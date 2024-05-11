from typing import Any, Optional

import japanize_matplotlib  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ._ot_utils import History, compute_nsw_loss, sinkhorn
from ._registry import register_optimizer
from .base import BaseClusteredOptimizer, BaseOptimizer

__all__ = ["PGDNSWOptimizer", "ClusteredPGDNSWOptimizer", "pgd_nsw", "clustered_pgd_nsw"]


def compute_pi_pgd_nsw(
    rel_mat: NDArray[np.float_],
    expo: NDArray[np.float_],
    high: NDArray[np.float_],
    alpha: float = 0.0,
    apply_negative_to_X_bf_sa: bool = True,
    eps: float = 0.01,
    lr: float = 0.01,
    max_iter: int = 200,
    ot_n_iter: int = 30,
    tol: float = 1e-6,
    device: str = "cpu",
    use_amp: Optional[bool] = None,
    pi_stock_size: int = 10,
) -> tuple[NDArray[np.float_], History]:
    """_description_

    Args:
        rel_mat (NDArray[np.float_]): relevance matrix. (n_query, n_doc)
        expo (NDArray[np.float_]): exposure matrix. (n_rank, 1)
        high (NDArray[np.float_]): high matrix. (n_doc, )
        alpha (float, optional): alpha. Defaults to 0.0.
        lr (float, optional): learning rate. Defaults to 0.01.
        ot_n_iter (int, optional): number of iteration for ot. Defaults to 30.
        last_ot_n_iter (int, optional): number of iteration for ot. Defaults to 100.
        tol (float, optional): tolerance. Defaults to 1e-6.
        device (str, optional): device. Defaults to "cpu".

    Returns:
        NDArray[np.float_]: _description_
    """
    n_query, n_doc = rel_mat.shape
    n_rank = expo.shape[0]
    rel_mat: torch.Tensor = torch.FloatTensor(rel_mat).to(device)  # (n_query, n_doc)
    expo: torch.Tensor = torch.FloatTensor(expo).to(device)  # (K, 1)
    am_rel = rel_mat.sum(0) ** alpha  # (n_doc, ), merit for each documnet, alpha nswに利用
    click_prob = rel_mat[:, :, None] * expo.reshape(1, 1, n_rank)  # (n_query, n_doc, K)

    # アイテムからの供給量
    a: torch.Tensor = torch.FloatTensor(high).view(1, -1, 1).to(device)
    # ダミー列への輸送量はアイテム数 - 表示数
    dummy_demand = a.sum(dim=1) - n_rank  # (n_query, 1)
    b = torch.ones(n_query, n_rank + 1, 1, device=device)
    b[:, -1, :] = dummy_demand

    # 初期値は一様分布
    # 表示数 < アイテム数の場合にはダミー列に輸送されるようにする
    X = nn.Parameter(torch.ones(n_query, n_doc, n_rank + 1).to(device) / n_doc)
    optimier = torch.optim.Adam([X], lr=lr)
    if use_amp is None:
        use_amp = True if device == "cuda" else False
    scaler = GradScaler(enabled=use_amp)
    history: History = History(pi_stock_size)
    if history.is_within_pi_stock_size:
        history.append_pi(X.data[0].clone().detach().cpu().numpy())

    for iter in tqdm(range(max_iter), leave=False):
        optimier.zero_grad()
        with autocast(enabled=use_amp):
            # 損失を計算
            loss = compute_nsw_loss(X[:, :, :-1], click_prob, am_rel)
        scaler.scale(loss).backward()
        scaler.step(optimier)
        scaler.update()

        # 収束判定のために勾配ノルムを計算
        grad_norm = torch.norm(X.grad)

        # シンクホーンアルゴリズムで実行可能領域に射影
        with autocast(enabled=use_amp):
            with torch.no_grad():
                if apply_negative_to_X_bf_sa:
                    X.data = -X.data
                X.data = sinkhorn(X, a, b, n_iter=ot_n_iter, eps=eps)

        # gradient normが一定以下になったら終了
        if grad_norm < tol:
            break

        history.append(loss.item(), grad_norm.item())
        if history.is_within_pi_stock_size:
            history.append_pi(X.data[0].clone().detach().cpu().numpy())

    pi: NDArray[np.float_] = X[:, :, :-1].detach().cpu().numpy()

    return pi, history


class PGDNSWOptimizer(BaseOptimizer):
    def __init__(
        self,
        alpha: float = 0.0,
        apply_negative_to_X_bf_sa: bool = True,
        eps: float = 0.01,
        lr: float = 0.01,
        max_iter: int = 200,
        ot_n_iter: int = 30,
        tol: float = 1e-6,
        device: str = "cpu",
        use_amp: Optional[bool] = None,
    ):
        self.alpha = alpha
        self.apply_negative_to_X_bf_sa = apply_negative_to_X_bf_sa
        self.eps = eps
        self.lr = lr
        self.max_iter = max_iter
        self.ot_n_iter = ot_n_iter
        self.tol = tol
        self.device = device
        self.use_amp = use_amp

    def solve(self, rel_mat: NDArray[np.float_], expo: NDArray[np.float_]) -> NDArray[np.float_]:
        n_doc = rel_mat.shape[1]
        high = np.ones(n_doc)
        pi, _ = compute_pi_pgd_nsw(
            rel_mat=rel_mat,
            expo=expo,
            high=high,
            alpha=self.alpha,
            apply_negative_to_X_bf_sa=self.apply_negative_to_X_bf_sa,
            eps=self.eps,
            lr=self.lr,
            max_iter=self.max_iter,
            ot_n_iter=self.ot_n_iter,
            tol=self.tol,
            device=self.device,
            use_amp=self.use_amp,
            pi_stock_size=0,
        )
        return pi


class ClusteredPGDNSWOptimizer(BaseClusteredOptimizer):
    def __init__(
        self,
        n_doc_cluster: int,
        n_query_cluster: int,
        alpha: float = 0.0,
        apply_negative_to_X_bf_sa: bool = True,
        eps: float = 1,
        lr: float = 0.01,
        max_iter: int = 200,
        ot_n_iter: int = 50,
        tol: float = 1e-6,
        device: str = "cpu",
        use_amp: Optional[bool] = None,
        random_state: int = 12345,
    ):
        super().__init__(n_doc_cluster, n_query_cluster, random_state)
        self.alpha = alpha
        self.apply_negative_to_X_bf_sa = apply_negative_to_X_bf_sa
        self.eps = eps
        self.lr = lr
        self.max_iter = max_iter
        self.ot_n_iter = ot_n_iter
        self.tol = tol
        self.device = device
        self.use_amp = use_amp

    def _solve(
        self, rel_mat: NDArray[np.float_], expo: NDArray[np.float_], high: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        pi, _ = compute_pi_pgd_nsw(
            rel_mat=rel_mat,
            expo=expo,
            high=high,
            alpha=self.alpha,
            apply_negative_to_X_bf_sa=self.apply_negative_to_X_bf_sa,
            eps=self.eps,
            lr=self.lr,
            max_iter=self.max_iter,
            ot_n_iter=self.ot_n_iter,
            tol=self.tol,
            device=self.device,
            use_amp=self.use_amp,
            pi_stock_size=0,
        )
        return pi


@register_optimizer
def pgd_nsw(**kwargs: Any) -> PGDNSWOptimizer:
    return PGDNSWOptimizer(**kwargs)


@register_optimizer
def clustered_pgd_nsw(**kwargs: Any) -> ClusteredPGDNSWOptimizer:
    return ClusteredPGDNSWOptimizer(**kwargs)
