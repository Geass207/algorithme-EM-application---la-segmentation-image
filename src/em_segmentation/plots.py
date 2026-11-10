"""Figures: fitted mixture over the intensity histogram, label map, convergence curve."""

from __future__ import annotations

import numpy as np

from em_segmentation.gmm import GMMFit
from em_segmentation.mri import TISSUES

COLORS = ("#3b6fb6", "#8a8f98", "#e8b04a", "#b5533c", "#5a9367", "#7d5ba6")  # CSF, gray, white, then extras
ROTATIONS = 2  # quarter turns applied for display, so that the top of the head is up


def _names(k: int) -> list[str]:
    return list(TISSUES) if k == len(TISSUES) else [f"Class {i + 1}" for i in range(k)]


def _gaussian(x, mean, variance):
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)


def plot_mixture(ax, pixels: np.ndarray, fit: GMMFit, bins: int = 60) -> None:
    ax.hist(pixels, bins=bins, density=True, color="#d5d8dc", label="Pixel intensities")
    x = np.linspace(pixels.min(), pixels.max(), 800)
    total = np.zeros_like(x)
    for k, name in enumerate(_names(len(fit.means))):
        y = fit.weights[k] * _gaussian(x, fit.means[k], fit.variances[k])
        total += y
        ax.plot(x, y, color=COLORS[k % len(COLORS)], linewidth=2, label=name)
    ax.plot(x, total, "k--", linewidth=1.5, label="Mixture")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)


def label_image(labels: np.ndarray) -> np.ndarray:
    """RGB image of a label map (black outside the brain)."""
    from matplotlib.colors import to_rgb

    rgb = np.zeros(labels.shape + (3,))
    for k in range(labels.max() + 1):
        rgb[labels == k] = to_rgb(COLORS[k % len(COLORS)])
    return rgb


def _show(ax, image, **kwargs):
    ax.imshow(np.rot90(image, k=ROTATIONS), **kwargs)
    ax.axis("off")


def summary_figure(image: np.ndarray, labels: np.ndarray, pixels: np.ndarray, fit: GMMFit):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    _show(axes[0], image, cmap="gray")
    axes[0].set_title("MRI slice")
    _show(axes[1], label_image(labels))
    axes[1].set_title("EM segmentation")
    plot_mixture(axes[2], pixels, fit)
    axes[2].set_title("Fitted Gaussian mixture")
    fig.tight_layout()
    return fig


def convergence_figure(fit: GMMFit):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(range(1, fit.n_iter + 1), fit.log_likelihoods, color="#3b6fb6", linewidth=2)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("The log-likelihood never decreases")
    fig.tight_layout()
    return fig


def iterations_figure(image, mask, pixels, snapshots: dict, valid):
    """Label map and mixture after selected iterations."""
    import matplotlib.pyplot as plt

    from em_segmentation.gmm import responsibilities

    fig, axes = plt.subplots(2, len(snapshots), figsize=(3.4 * len(snapshots), 6.2))
    for col, (iteration, (weights, means, variances)) in enumerate(snapshots.items()):
        order = np.argsort(means)
        fit = GMMFit(weights[order], means[order], variances[order],
                     responsibilities(pixels, weights[order], means[order], variances[order]))
        labels = np.full(image.shape, -1, dtype=int)
        labels[valid] = fit.labels
        _show(axes[0, col], label_image(labels))
        axes[0, col].set_title(f"Iteration {iteration}")
        plot_mixture(axes[1, col], pixels, fit)
        axes[1, col].get_legend().remove()
    fig.tight_layout()
    return fig
