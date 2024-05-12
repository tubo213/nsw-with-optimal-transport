from dataclasses import dataclass

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from numpy.typing import NDArray


def sinkhorn(
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    n_iter: int = 15,
    eps: float = 0.1,
    apply_kernel: bool = True,
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
    if apply_kernel:
        K = torch.exp(-C / eps)  # (n_query, n_doc, n_rank + 1)
    else:
        K = C
    u = torch.ones(n_query, n_doc, 1, device=C.device, dtype=C.dtype)  # (n_query, n_doc, 1)

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


@dataclass
class History:
    """
    A class to store and visualize optimization history.

    Attributes:
        pi_stock_size (int): The maximum number of pi values to store.
        grad_stock_size (int): The maximum number of gradient values to store.

    Methods:
        is_within_pi_stock_size: Check if the number of stored pi values is within the limit.
        append: Append a loss and gradient norm value to the history.
        append_pi: Append a pi value to the history.
        plot_loss_curve: Plot the loss curve.
        plot_grad_norm_curve: Plot the gradient norm curve.
        plot_pi_by_iteration: Plot the pi values by iteration.

    """

    pi_stock_size: int = 0
    grad_stock_size: int = 0

    def __post_init__(self) -> None:
        self.loss: list[tuple[int, float]] = []
        self.grad_norm: list[tuple[int, float]] = []
        self.pi: list[tuple[int, NDArray[np.float_]]] = []
        self.grad: list[tuple[int, NDArray[np.float_]]] = []

    @property
    def is_within_pi_stock_size(self) -> bool:
        """
        Check if the number of stored pi values is within the limit.

        Returns:
            bool: True if the number of stored pi values is within the limit, False otherwise.

        """
        return len(self.pi) < self.pi_stock_size

    @property
    def is_within_grad_stock_size(self) -> bool:
        """
        Check if the number of stored gradient values is within the limit.

        Returns:
            bool: True if the number of stored gradient values is within the limit, False otherwise.

        """
        return len(self.grad) < self.grad_stock_size

    def append_loss(self, iter: int, loss: float) -> None:
        self.loss.append((iter, loss))

    def append_grad_norm(self, iter: int, grad_norm: float) -> None:
        self.grad_norm.append((iter, grad_norm))

    def append_pi_if_not_full(self, iter: int, pi: NDArray[np.float_]) -> None:
        if self.is_within_pi_stock_size:
            self.pi.append((iter, pi))

    def append_grad_if_not_full(self, iter: int, grad: NDArray[np.float_]) -> None:
        if self.is_within_grad_stock_size:
            self.grad.append((iter, grad))

    def plot_loss_curve(self) -> None:
        """
        Plot the loss curve.

        Returns:
            None

        """
        plt.figure()
        iters, loss = zip(*self.loss)
        plt.plot(iters, loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

    def plot_grad_norm_curve(self) -> None:
        """
        Plot the gradient norm curve.

        Returns:
            None

        """
        plt.figure()
        iters, grad_norm = zip(*self.grad_norm)
        plt.plot(iters, grad_norm)
        plt.xlabel("Iteration")
        plt.ylabel("Grad norm")

    def plot_pi_by_iteration(self, n_col: int = 5, plot_dummy: bool = True) -> None:
        """
        Plot the pi values by iteration.

        Args:
            n_col (int): The number of columns in the subplot grid. Default is 5.
            plot_dummy (bool): Whether to plot the dummy column in the pi values. Default is True.

        Returns:
            None

        """
        n_row = len(self.pi) // n_col
        if (len(self.pi) % n_col != 0) or (n_row == 0):
            n_row += 1
        fig, axes = plt.subplots(n_row, n_col, figsize=(5 * n_col, 5 * n_row))
        axes: list[Axes] = np.ravel(axes).tolist()
        for i, (iter, pi) in enumerate(self.pi):
            ax = axes[i]
            if not plot_dummy:
                pi = pi[:, :-1]
            sns.heatmap(
                pi,
                annot=False,
                fmt=".1f",
                cmap="Blues",
                ax=ax,
                vmin=0,
                vmax=1,
                cbar=True if i == 0 else False,
            )
            ax.set_title(f"表示確率: イテレーション {iter}")
            ax.set_xlabel("表示位置")
            ax.set_ylabel("アイテム")
            if plot_dummy:
                # xメモリの一番最後をdummyに変更
                ax.set_xticklabels([f"{i}" for i in range(pi.shape[1] - 1)] + ["dummy"])

        fig.tight_layout()

    def plot_grad_by_iteration(self, n_col: int = 5) -> None:
        """
        Plot the gradient values by iteration.

        Args:
            n_col (int): The number of columns in the subplot grid. Default is 5.

        Returns:
            None

        """
        n_row = len(self.grad) // n_col
        if (len(self.grad) % n_col != 0) or (n_row == 0):
            n_row += 1
        fig, axes = plt.subplots(n_row, n_col, figsize=(5 * n_col, 5 * n_row))
        axes: list[Axes] = np.ravel(axes).tolist()
        for i, (iter, grad) in enumerate(self.grad):
            ax = axes[i]
            sns.heatmap(
                grad[:, :-1],
                annot=False,
                fmt=".1f",
                cmap="Blues",
                ax=ax,
                cbar=True,
            )
            ax.set_title(f"勾配: イテレーション {iter}")
            ax.set_xlabel("表示位置")
            ax.set_ylabel("アイテム")

        fig.tight_layout()
