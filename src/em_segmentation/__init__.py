"""EM for Gaussian mixtures, applied to brain-MRI tissue segmentation."""

from em_segmentation.gmm import GMMFit, bic, fit_gmm, log_likelihood, responsibilities

__all__ = ["GMMFit", "bic", "fit_gmm", "log_likelihood", "responsibilities"]
