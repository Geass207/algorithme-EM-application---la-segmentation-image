"""Command line: segment one slice of an MRI volume and save the figures.

    python -m em_segmentation.segment --image data/brain.nii.gz --mask data/brain_mask.nii
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from em_segmentation.mri import AXES, TISSUES, load_volume, segment_slice, take_slice


def main() -> None:
    parser = argparse.ArgumentParser(description="EM tissue segmentation of a brain-MRI slice")
    parser.add_argument("--image", required=True, help="NIfTI volume")
    parser.add_argument("--mask", required=True, help="NIfTI brain mask of the same shape")
    parser.add_argument("--view", choices=sorted(AXES), default="coronal")
    parser.add_argument("--slice", type=int, default=None, help="slice index (default: middle)")
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--min-intensity", type=float, default=100.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from em_segmentation import plots

    image = take_slice(load_volume(args.image), args.view, args.slice)
    mask = take_slice(load_volume(args.mask), args.view, args.slice)

    snapshots = {}
    keep = (1, 10, 50)

    def record(iteration, weights, means, variances):
        if iteration in keep:
            snapshots[iteration] = (weights.copy(), means.copy(), variances.copy())

    labels, fit, valid = segment_slice(
        image, mask, n_components=args.classes, min_intensity=args.min_intensity,
        max_iter=args.max_iter, tol=args.tol, callback=record,
    )
    pixels = image[valid]
    snapshots[fit.n_iter] = (fit.weights, fit.means, fit.variances)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots.summary_figure(image, labels, pixels, fit).savefig(out / "segmentation.png", dpi=160)
    plots.convergence_figure(fit).savefig(out / "log_likelihood.png", dpi=160)
    plots.iterations_figure(image, mask, pixels, snapshots, valid).savefig(out / "iterations.png", dpi=140)

    names = TISSUES if args.classes == len(TISSUES) else [f"class {k + 1}" for k in range(args.classes)]
    summary = {
        "pixels": int(pixels.size),
        "iterations": fit.n_iter,
        "converged": fit.converged,
        "log_likelihood": fit.log_likelihoods[-1],
        "classes": [
            {"name": name, "weight": float(w), "mean": float(m), "std": float(np.sqrt(v)),
             "pixels": int((fit.labels == k).sum())}
            for k, (name, w, m, v) in enumerate(zip(names, fit.weights, fit.means, fit.variances))
        ],
    }
    from em_segmentation.gmm import bic, fit_gmm

    selection = []
    for k in range(1, 7):
        candidate = fit_gmm(pixels, n_components=k, max_iter=5000, tol=1e-10)
        selection.append({
            "classes": k, "iterations": candidate.n_iter, "converged": candidate.converged,
            "log_likelihood": candidate.log_likelihoods[-1],
            "bic": bic(candidate.log_likelihoods[-1], k, pixels.size),
            "means": [float(m) for m in candidate.means],
        })
    summary["model_selection"] = selection
    (out / "parameters.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
