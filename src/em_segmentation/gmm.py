"""Expectation-Maximization for a one-dimensional Gaussian mixture model.

Model: each observation x_i has a hidden class z_i in {1..K} with
p(z_i = k) = pi_k and x_i | z_i = k ~ N(mu_k, sigma_k^2).

EM maximises the log-likelihood through the lower bound obtained with Jensen's
inequality: for any distributions q_i over the classes,

    log p(x; theta) >= sum_i sum_k q_ik log( pi_k N(x_i; mu_k, sigma_k^2) / q_ik ).

The E-step makes the bound tight by setting q_ik to the posterior p(z_i = k | x_i);
the M-step maximises it over theta in closed form. Each iteration therefore
cannot decrease the log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class GMMFit:
    weights: np.ndarray          # pi_k, shape (K,)
    means: np.ndarray            # mu_k, shape (K,)
    variances: np.ndarray        # sigma_k^2, shape (K,)
    responsibilities: np.ndarray  # q_ik, shape (n, K)
    log_likelihoods: list[float] = field(default_factory=list)  # one value per iteration
    converged: bool = False

    @property
    def n_iter(self) -> int:
        return len(self.log_likelihoods)

    @property
    def labels(self) -> np.ndarray:
        """Most probable class of each observation."""
        return np.argmax(self.responsibilities, axis=1)


def _log_joint(x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """log( pi_k N(x_i; mu_k, sigma_k^2) ), shape (n, K)."""
    diff = x[:, None] - means[None, :]
    return np.log(weights)[None, :] - 0.5 * (LOG_2PI + np.log(variances)[None, :] + diff**2 / variances[None, :])


def _logsumexp(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))).ravel()


def log_likelihood(x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> float:
    return float(_logsumexp(_log_joint(np.asarray(x, dtype=float), weights, means, variances)).sum())


def bic(log_lik: float, n_components: int, n_observations: int) -> float:
    """Bayesian information criterion; a 1-D mixture of K Gaussians has 3K - 1 free parameters."""
    return -2.0 * log_lik + (3 * n_components - 1) * np.log(n_observations)


def responsibilities(x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """E-step: posterior probability of each class for each observation."""
    log_joint = _log_joint(np.asarray(x, dtype=float), weights, means, variances)
    return np.exp(log_joint - _logsumexp(log_joint)[:, None])


def initial_parameters(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic start: equal weights, means spread evenly over the data range,
    and a common standard deviation equal to the spacing between the means."""
    lo, hi = float(np.min(x)), float(np.max(x))
    step = (hi - lo) / n_components
    means = lo + step * np.arange(1, n_components + 1)
    return np.full(n_components, 1.0 / n_components), means, np.full(n_components, step**2)


def fit_gmm(
    x: np.ndarray,
    n_components: int = 3,
    max_iter: int = 500,
    tol: float = 1e-8,
    min_variance: float = 1e-6,
    init: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    callback=None,
) -> GMMFit:
    """Fit a 1-D Gaussian mixture by EM.

    Stops when the relative increase of the log-likelihood falls below ``tol``.
    ``min_variance`` keeps a component from collapsing onto a single point, where
    the likelihood is unbounded. ``callback(iteration, weights, means, variances)``
    is called after every M-step.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < n_components:
        raise ValueError("fewer observations than mixture components")
    weights, means, variances = init if init is not None else initial_parameters(x, n_components)
    weights, means, variances = (np.array(a, dtype=float) for a in (weights, means, variances))

    history: list[float] = []
    converged = False
    for iteration in range(1, max_iter + 1):
        # E-step
        log_joint = _log_joint(x, weights, means, variances)
        log_norm = _logsumexp(log_joint)
        q = np.exp(log_joint - log_norm[:, None])
        history.append(float(log_norm.sum()))  # log-likelihood of the current parameters

        # M-step: weighted maximum-likelihood estimates
        n_k = q.sum(axis=0)
        weights = n_k / x.size
        means = (q * x[:, None]).sum(axis=0) / n_k
        variances = np.maximum((q * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / n_k, min_variance)
        if callback is not None:
            callback(iteration, weights, means, variances)

        if len(history) > 1 and history[-1] - history[-2] <= tol * abs(history[-2]):
            converged = True
            break

    order = np.argsort(means)  # report classes by increasing mean
    weights, means, variances = weights[order], means[order], variances[order]
    return GMMFit(weights, means, variances, responsibilities(x, weights, means, variances), history, converged)
