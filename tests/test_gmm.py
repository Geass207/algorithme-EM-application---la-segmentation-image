import numpy as np
import pytest

from em_segmentation.gmm import fit_gmm, initial_parameters, log_likelihood, responsibilities
from em_segmentation.mri import segment_slice


def sample(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    weights, means, stds = np.array([0.2, 0.5, 0.3]), np.array([100.0, 300.0, 520.0]), np.array([25.0, 40.0, 30.0])
    z = rng.choice(3, size=n, p=weights)
    return rng.normal(means[z], stds[z]), z, weights, means, stds


def test_em_recovers_the_mixture_parameters():
    x, _, weights, means, stds = sample()
    fit = fit_gmm(x, n_components=3)
    assert fit.converged
    assert fit.weights == pytest.approx(weights, abs=0.03)
    assert fit.means == pytest.approx(means, rel=0.03)
    assert np.sqrt(fit.variances) == pytest.approx(stds, rel=0.12)


def test_log_likelihood_never_decreases():
    x, *_ = sample(seed=1)
    history = fit_gmm(x, n_components=3, max_iter=200).log_likelihoods
    assert all(b >= a - 1e-9 * abs(a) for a, b in zip(history, history[1:]))
    assert history[-1] > history[0]


def test_responsibilities_are_posterior_probabilities():
    x, *_ = sample(n=500)
    q = responsibilities(x, *initial_parameters(x, 3))
    assert q.shape == (500, 3)
    assert q.sum(axis=1) == pytest.approx(np.ones(500))
    assert (q >= 0).all()


def test_fit_matches_reported_log_likelihood_and_orders_classes():
    x, *_ = sample(seed=2)
    fit = fit_gmm(x, n_components=3)
    assert (np.diff(fit.means) > 0).all()
    assert log_likelihood(x, fit.weights, fit.means, fit.variances) >= fit.log_likelihoods[-1]


def test_most_points_get_their_true_class():
    x, z, *_ = sample(seed=3)
    assert (fit_gmm(x, n_components=3).labels == z).mean() > 0.93


def test_segment_slice_ignores_pixels_outside_the_mask():
    rng = np.random.default_rng(0)
    image = np.zeros((40, 40))
    image[:, :20] = rng.normal(200, 10, (40, 20))
    image[:, 20:] = rng.normal(600, 10, (40, 20))
    mask = np.ones_like(image)
    mask[:5] = 0
    labels, fit, valid = segment_slice(image, mask, n_components=2, min_intensity=50)
    assert (labels[:5] == -1).all() and valid.sum() == 35 * 40
    assert (labels[5:, :20] == 0).all() and (labels[5:, 20:] == 1).all()
    assert fit.means == pytest.approx([200, 600], abs=3)
