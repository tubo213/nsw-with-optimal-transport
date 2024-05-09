from typing import Any, Literal, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.cuda.amp import GradScaler, autocast  # type: ignore

from ._registry import register_optimizer
from .base import BaseClusteredOptimizer, BaseOptimizer

__all__ = ["OTNSWOptimizer", "ClusteredOTNSWOptimizer", "ot_nsw", "clustered_ot_nsw"]

METHOD = Literal["ot", "pg_ot"]


def sinkhorn(
    C: torch.Tensor, a: Union[int, torch.Tensor] = 1, n_iter: int = 15, eps: float = 0.1
) -> torch.Tensor:
    """
    Applies the Sinkhorn algorithm to compute optimal transport between two sets of points.

    Args:
        C (torch.Tensor): Cost matrix. Shape: (n_query, n_doc, n_rank)
        a (Union[int, torch.Tensor], optional): Regularization parameter. Shape: (n_query, n_doc, 1). Defaults to 1.
        n_iter (int, optional): Number of iterations. Defaults to 15.
        eps (float, optional): Epsilon parameter. Defaults to 0.1.

    Returns:
        torch.Tensor: Optimal transport matrix. Shape: (n_query, n_doc, n_rank)
    """
    device = C.device
    n_query, n_doc, _ = C.shape
    K = torch.exp(-C / eps)  # (n_query, n_doc, n_rank)
    u = torch.ones(n_query, n_doc, 1, device=device)

    # sinkhorn iteration
    for _ in range(n_iter):
        v: torch.Tensor = 1 / (torch.bmm(K.transpose(1, 2), u)).detach()  # (n_query, n_doc, 1)
        u = a / (torch.bmm(K, v)).detach()  # (n_query, n_doc, 1)

    return u * K * v.transpose(1, 2)


def normalize_a(a: torch.Tensor, high: torch.Tensor, k: int) -> torch.Tensor:
    """_description_

    Args:
        a (torch.Tensor): (n_query, n_doc, 1)
        high (torch.Tensor): (1, n_doc, 1)

    Returns:
        torch.Tensor: (n_query, n_doc, 1)
    """
    low = torch.zeros_like(high).to(a.device)
    a_hat = (a / a.sum(dim=1, keepdim=True)) * k
    a_hat = torch.clamp(a_hat, low, high)
    return a_hat


def compute_nsw_loss(
    pi: torch.Tensor,
    click_prob: torch.Tensor,
    am_rel: torch.Tensor,
) -> torch.Tensor:
    """

    Args:
        pi (torch.Tensor): probability matrix. (n_query, n_doc, n_rank)
        click_prob (torch.Tensor): click probability matrix. (n_query, n_doc, n_rank)
        am_rel (torch.Tensor):(n_doc)
    """
    imp = (pi * click_prob).sum(dim=[0, 2])
    return -(am_rel * torch.log(imp)).sum()


def compute_pi_ot_nsw(
    rel_mat: NDArray[np.float_],
    expo: NDArray[np.float_],
    high: NDArray[np.float_],
    alpha: float = 0.0,
    lr: float = 0.01,
    ot_n_iter: int = 30,
    last_ot_n_iter: int = 100,
    tol: float = 1e-6,
    device: str = "cpu",
) -> NDArray[np.float_]:
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
    rel_tensor = torch.FloatTensor(rel_mat).to(device)  # (n_query, n_doc)
    expo_tensor = torch.FloatTensor(expo).to(device)  # (K, 1)
    high_tensor = torch.FloatTensor(high).view(1, -1, 1).to(device)
    am_rel = rel_tensor.sum(0) ** alpha  # (n_doc, ), merit for each documnet, alpha nswに利用
    click_prob = rel_tensor[:, :, None] * expo_tensor.reshape(1, 1, n_rank)  # (n_query, n_doc, K)

    # optimization
    C = nn.Parameter(torch.rand(n_query, n_doc, n_rank).to(device))
    a = nn.Parameter(torch.ones(n_query, n_doc, 1).to(device))
    optimier = torch.optim.Adam([C, a], lr=lr)
    use_amp = True if device == "cuda" else False
    scaler = GradScaler(enabled=use_amp)
    prev_loss = 1e7
    while True:
        optimier.zero_grad()
        with autocast(enabled=use_amp):
            # project to feasible set
            C.data = torch.clamp_(C.data, 0.0)
            a.data = torch.clamp_(a.data, 0.0)
            # normalize a
            a_hat = normalize_a(a, high_tensor, n_rank)

            # compute pi
            pi: torch.Tensor = sinkhorn(C, a_hat, n_iter=ot_n_iter)
            loss = compute_nsw_loss(pi, click_prob, am_rel)
        scaler.scale(loss).backward()
        scaler.step(optimier)
        scaler.update()

        diff = abs(prev_loss - loss.item())
        if diff < tol:
            break
        prev_loss = loss.item()

    # compute last pi
    with torch.no_grad():
        with autocast(enabled=use_amp):
            # normalize a
            a_hat = normalize_a(a, high_tensor, n_rank)
            # compute pi
            pi = sinkhorn(C, a_hat, n_iter=last_ot_n_iter)
            pi /= pi.sum(dim=1, keepdim=True)

    return pi.detach().cpu().numpy()  # type: ignore


def compute_pi_pg_ot_nsw(
    rel_mat: NDArray[np.float_],
    expo: NDArray[np.float_],
    high: NDArray[np.float_],
    alpha: float = 0.0,
    lr: float = 0.01,
    ot_n_iter: int = 30,
    last_ot_n_iter: int = 100,
    tol: float = 1e-6,
    device: str = "cpu",
) -> NDArray[np.float_]:
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
    rel_tensor = torch.FloatTensor(rel_mat).to(device)  # (n_query, n_doc)
    expo_tensor = torch.FloatTensor(expo).to(device)  # (K, 1)
    high_tensor = torch.FloatTensor(high).view(1, -1, 1).to(device)
    am_rel = rel_tensor.sum(0) ** alpha  # (n_doc, ), merit for each documnet, alpha nswに利用
    click_prob = rel_tensor[:, :, None] * expo_tensor.reshape(1, 1, n_rank)  # (n_query, n_doc, K)

    # optimization
    X = nn.Parameter(torch.rand(n_query, n_doc, n_rank).to(device))
    a = nn.Parameter(torch.ones(n_query, n_doc, 1).to(device))
    optimier = torch.optim.Adam([X, a], lr=lr)
    use_amp = True if device == "cuda" else False
    scaler = GradScaler(enabled=use_amp)
    prev_loss = 1e7
    while True:
        optimier.zero_grad()
        with autocast(enabled=use_amp):
            # project to feasible set
            X.data = torch.clamp_(X.data, 0.0)
            a.data = torch.clamp_(a.data, 0.0)
            a_hat = normalize_a(a, high_tensor, n_rank)
            with torch.no_grad():
                X.data = sinkhorn(X, a_hat, n_iter=ot_n_iter)

            # compute loss
            loss = compute_nsw_loss(X, click_prob, am_rel)
        scaler.scale(loss).backward()
        scaler.step(optimier)
        scaler.update()

        diff = abs(prev_loss - loss.item())
        if diff < tol:
            break
        prev_loss = loss.item()

    # compute last pi
    with torch.no_grad():
        with autocast(enabled=use_amp):
            # normalize a
            a_hat = normalize_a(a, high_tensor, n_rank)
            # compute pi
            X.data = sinkhorn(X, a_hat, n_iter=last_ot_n_iter)

    X: NDArray[np.float_] = X.detach().cpu().numpy()
    return X


class OTNSWOptimizer(BaseOptimizer):
    def __init__(
        self,
        method: METHOD = "ot",
        alpha: float = 0.0,
        lr: float = 0.01,
        ot_n_iter: int = 50,
        last_ot_n_iter: int = 100,
        tol: float = 1e-6,
        device: str = "cpu",
    ):
        self.method = method
        self.alpha = alpha
        self.lr = lr
        self.ot_n_iter = ot_n_iter
        self.last_ot_n_iter = last_ot_n_iter
        self.tol = tol
        self.device = device

    def solve(self, rel_mat: NDArray[np.float_], expo: NDArray[np.float_]) -> NDArray[np.float_]:
        n_doc = rel_mat.shape[1]
        high = np.ones(n_doc)
        if self.method == "ot":
            return compute_pi_ot_nsw(
                rel_mat,
                expo,
                high,
                self.alpha,
                self.lr,
                self.ot_n_iter,
                self.last_ot_n_iter,
                self.tol,
                self.device,
            )
        elif self.method == "pg_ot":
            return compute_pi_pg_ot_nsw(
                rel_mat,
                expo,
                high,
                self.alpha,
                self.lr,
                self.ot_n_iter,
                self.last_ot_n_iter,
                self.tol,
                self.device,
            )


class ClusteredOTNSWOptimizer(BaseClusteredOptimizer):
    def __init__(
        self,
        n_doc_cluster: int,
        n_query_cluster: int,
        method: METHOD = "ot",
        alpha: float = 0.0,
        lr: float = 0.01,
        ot_n_iter: int = 50,
        last_ot_n_iter: int = 100,
        tol: float = 1e-6,
        device: str = "cpu",
        random_state: int = 12345,
    ):
        super().__init__(n_doc_cluster, n_query_cluster, random_state)
        self.method = method
        self.alpha = alpha
        self.lr = lr
        self.ot_n_iter = ot_n_iter
        self.last_ot_n_iter = last_ot_n_iter
        self.tol = tol
        self.device = device

    def _solve(
        self, rel_mat: NDArray[np.float_], expo: NDArray[np.float_], high: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        if self.method == "ot":
            return compute_pi_ot_nsw(
                rel_mat,
                expo,
                high,
                self.alpha,
                self.lr,
                self.ot_n_iter,
                self.last_ot_n_iter,
                self.tol,
                self.device,
            )
        elif self.method == "pg_ot":
            return compute_pi_pg_ot_nsw(
                rel_mat,
                expo,
                high,
                self.alpha,
                self.lr,
                self.ot_n_iter,
                self.last_ot_n_iter,
                self.tol,
                self.device,
            )


@register_optimizer
def ot_nsw(**kwargs: Any) -> OTNSWOptimizer:
    return OTNSWOptimizer(**kwargs)


@register_optimizer
def clustered_ot_nsw(**kwargs: Any) -> ClusteredOTNSWOptimizer:
    return ClusteredOTNSWOptimizer(**kwargs)
