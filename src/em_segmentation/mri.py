"""Loading an MRI volume, extracting a slice and turning a mixture fit into a label map."""

from __future__ import annotations

import numpy as np

from em_segmentation.gmm import GMMFit, fit_gmm

# On a T1-weighted image, mean intensity increases from cerebrospinal fluid to gray to white matter.
TISSUES = ("Cerebrospinal fluid", "Gray matter", "White matter")
AXES = {"sagittal": 0, "coronal": 1, "axial": 2}


def load_volume(path: str) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(path).get_fdata())


def take_slice(volume: np.ndarray, view: str = "coronal", index: int | None = None) -> np.ndarray:
    """2-D section of a volume; the middle one by default."""
    axis = AXES[view]
    index = volume.shape[axis] // 2 if index is None else index
    return np.take(volume, index, axis=axis)


def segment_slice(
    image: np.ndarray, mask: np.ndarray, n_components: int = 3, min_intensity: float = 100.0, **fit_kwargs
) -> tuple[np.ndarray, GMMFit, np.ndarray]:
    """Fit the mixture on brain pixels and label each of them.

    Pixels outside the brain mask, or darker than ``min_intensity`` (background
    and partial-volume pixels at the brain boundary), are excluded.

    Returns ``(labels, fit, valid)``: ``labels`` has the shape of the image, with
    -1 outside the brain and 0..K-1 inside, ordered by increasing mean intensity.
    """
    valid = (mask > 0) & (image > min_intensity)
    if not valid.any():
        raise ValueError("no pixel left after masking and thresholding")
    fit = fit_gmm(image[valid], n_components=n_components, **fit_kwargs)
    labels = np.full(image.shape, -1, dtype=int)
    labels[valid] = fit.labels
    return labels, fit, valid
