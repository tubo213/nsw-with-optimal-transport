from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ._registry import register_optimizer
from .base import BaseClusteredOptimizer, BaseOptimizer

__all__ = ["OTNSWOptimizer", "ClusteredOTNSWOptimizer", "ot_nsw", "clustered_ot_nsw"]

METHOD = Literal["ot", "pg_ot"]


def sinkhorn(
    C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, n_iter: int = 15, eps: float = 0.1
) -> torch.Tensor:
    """
    Applies the Sinkhorn algorithm to compute optimal transport between two sets of points.

    Args:
        C (torch.Tensor): Cost matrix. Shape: (n_query, n_doc, n_rank + 1)
        a (torch.Tensor]): Regularization parameter. Shape: (n_query, n_doc, 1). Defaults to 1.
        b (torch.Tensor]): Regularization parameter. Shape: (n_query, n_rank + 1, 1). Defaults to 1.
        n_iter (int, optional): Number of iterations. Defaults to 15.
        eps (float, optional): Epsilon parameter. Defaults to 0.1.

    Returns:
        torch.Tensor: Optimal transport matrix. Shape: (n_query, n_doc, n_rank + 1)
    """
    n_query, n_doc, _ = C.shape
    K = torch.exp(-C / eps)  # (n_query, n_doc, n_rank + 1)
    u = torch.ones(n_query, n_doc, 1, device=C.device)  # (n_query, n_doc, 1)

    # sinkhorn iteration
    for _ in range(n_iter):
        v = b / (torch.bmm(K.transpose(1, 2), u))  # (n_query, n_rank + 1, 1)
        u = a / (torch.bmm(K, v))  # (n_query, n_doc, 1)

    # 数値誤差で列和が1を超えることがあるため最後に列和を正規化
    # 行和はダミー列があるため1を超えることはない
    v = b / (torch.bmm(K.transpose(1, 2), u))

    return u * K * v.transpose(1, 2)  # (n_query, n_doc, n_rank + 1)


def compute_nsw_loss(
    pi: torch.Tensor,
    click_prob: torch.Tensor,
    am_rel: torch.Tensor,
    eps: float = 0,
) -> torch.Tensor:
    """
    Compute the Nash Social Welfare (NSW) loss.

    Args:
        pi (torch.Tensor): Probability matrix of shape (n_query, n_doc, n_rank).
        click_prob (torch.Tensor): Click probability matrix of shape (n_query, n_doc, n_rank).
        am_rel (torch.Tensor): Relevance matrix of shape (n_doc).
        eps (float, optional): Small value added to the denominator to avoid division by zero. Defaults to 0.

    Returns:
        torch.Tensor: The computed NSW loss.

    """
    imp = (pi * click_prob).sum(dim=[0, 2])
    return -(am_rel * torch.log(imp + eps)).sum()


def compute_pi_ot_nsw(
    rel_mat: NDArray[np.float_],
    expo: NDArray[np.float_],
    high: NDArray[np.float_],
    alpha: float = 0.0,
    eps: float = 0.01,
    lr: float = 0.01,
    max_iter: int = 200,
    ot_n_iter: int = 30,
    tol: float = 1e-6,
    device: str = "cpu",
    use_amp: Optional[bool] = None,
) -> tuple[NDArray[np.float_], dict[str, list[float]]]:
    """_description_

    Args:
        rel_mat (NDArray[np.float_]): relevance matrix. (n_query, n_doc)
        expo (NDArray[np.float_]): exposure matrix. (n_rank, 1)
        high (NDArray[np.float_]): high matrix. (n_doc, )
        alpha (float, optional): alpha. Defaults to 0.0.
        lr (float, optional): learning rate. Defaults to 0.01.
        ot_n_iter (int, optional): number of iteration for ot. Defaults to 30.
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
    C = nn.Parameter(torch.ones(n_query, n_doc, n_rank + 1).to(device))
    optimier = torch.optim.Adam([C], lr=lr)
    if use_amp is None:
        use_amp = True if device == "cuda" else False
    scaler = GradScaler(enabled=use_amp)
    history: dict[str, list[float]] = {
        "loss": [],
        "grad_norm": [],
    }
    for _ in tqdm(range(max_iter)):
        optimier.zero_grad()
        with autocast(enabled=use_amp):
            # compute pi
            X: torch.Tensor = sinkhorn(C, a, b, n_iter=ot_n_iter, eps=eps)
            loss = compute_nsw_loss(X[:, :, :-1], click_prob, am_rel)
        scaler.scale(loss).backward()
        scaler.step(optimier)
        scaler.update()

        # gradient normが一定以下になったら終了
        grad_norm = torch.norm(C.grad)
        if grad_norm < tol:
            break

        history["loss"].append(loss.item())
        history["grad_norm"].append(grad_norm.item())

    pi: NDArray[np.float_] = X[:, :, :-1].detach().cpu().numpy()

    return pi, history


def compute_pi_pg_ot_nsw(
    rel_mat: NDArray[np.float_],
    expo: NDArray[np.float_],
    high: NDArray[np.float_],
    alpha: float = 0.0,
    eps: float = 0.01,
    lr: float = 0.01,
    max_iter: int = 200,
    ot_n_iter: int = 30,
    tol: float = 1e-6,
    device: str = "cpu",
    use_amp: Optional[bool] = None,
) -> tuple[NDArray[np.float_], dict[str, list[float]]]:
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
    X = nn.Parameter(torch.rand(n_query, n_doc, n_rank + 1).to(device) / n_doc)
    optimier = torch.optim.Adam([X], lr=lr)
    if use_amp is None:
        use_amp = True if device == "cuda" else False
    scaler = GradScaler(enabled=use_amp)
    history: dict[str, list[float]] = {
        "loss": [],
        "grad_norm": [],
    }

    for iter in tqdm(range(max_iter)):
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
                X.data = sinkhorn((-X), a, b, n_iter=ot_n_iter, eps=eps)

        # gradient normが一定以下になったら終了
        if grad_norm < tol:
            break

        history["loss"].append(loss.item())
        history["grad_norm"].append(grad_norm.item())

    pi: NDArray[np.float_] = X[:, :, :-1].detach().cpu().numpy()

    return pi, history


class OTNSWOptimizer(BaseOptimizer):
    def __init__(
        self,
        method: METHOD = "ot",
        alpha: float = 0.0,
        eps: float = 1,
        lr: float = 0.01,
        max_iter: int = 200,
        ot_n_iter: int = 50,
        tol: float = 1e-6,
        device: str = "cpu",
        use_amp: Optional[bool] = None,
    ):
        self.method = method
        self.alpha = alpha
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
        if self.method == "ot":
            pi, _ = compute_pi_ot_nsw(
                rel_mat,
                expo,
                high,
                self.alpha,
                self.eps,
                self.lr,
                self.max_iter,
                self.ot_n_iter,
                self.tol,
                self.device,
                self.use_amp,
            )
            return pi
        elif self.method == "pg_ot":
            pi, _ = compute_pi_pg_ot_nsw(
                rel_mat,
                expo,
                high,
                self.alpha,
                self.eps,
                self.lr,
                self.max_iter,
                self.ot_n_iter,
                self.tol,
                self.device,
            )
            return pi
        else:
            raise ValueError(f"Invalid method: {self.method}")


class ClusteredOTNSWOptimizer(BaseClusteredOptimizer):
    def __init__(
        self,
        n_doc_cluster: int,
        n_query_cluster: int,
        method: METHOD = "ot",
        alpha: float = 0.0,
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
        self.method = method
        self.alpha = alpha
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
        if self.method == "ot":
            pi, _ = compute_pi_ot_nsw(
                rel_mat,
                expo,
                high,
                self.alpha,
                self.eps,
                self.lr,
                self.max_iter,
                self.ot_n_iter,
                self.tol,
                self.device,
                self.use_amp,
            )
            return pi
        elif self.method == "pg_ot":
            pi, _ = compute_pi_pg_ot_nsw(
                rel_mat,
                expo,
                high,
                self.alpha,
                self.eps,
                self.lr,
                self.max_iter,
                self.ot_n_iter,
                self.tol,
                self.device,
                self.use_amp,
            )
            return pi
        else:
            raise ValueError(f"Invalid method: {self.method}")


@register_optimizer
def ot_nsw(**kwargs: Any) -> OTNSWOptimizer:
    return OTNSWOptimizer(**kwargs)


@register_optimizer
def clustered_ot_nsw(**kwargs: Any) -> ClusteredOTNSWOptimizer:
    return ClusteredOTNSWOptimizer(**kwargs)
