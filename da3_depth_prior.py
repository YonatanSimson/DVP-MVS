#!/usr/bin/env python
"""Generate DVP-MVS's `dep/%08d.dmb` monocular depth-prior files using Depth Anything 3.

DVP-MVS's FIRST_INIT stage (APD.cpp) reads `<dense_folder>/dep/%08d.dmb` expecting a
disparity-like map (near = high value, roughly in [0, 255]) matching the working image's
own resolution -- it then does `255 - dep` to flip it depth-proportional, and rescales it
to metric depth using sparse COLMAP points via `<dense_folder>/sfm/%08d.txt`
(written by colmap2mvsnet.py's write_sfm_correspondences). Run this against the
`images/` folder colmap2mvsnet.py already produced, so filenames/ordering line up with
`cams/`.

Usage:
    python da3_depth_prior.py --dense_folder <mvsnet_format_dir> \
        [--model depth-anything/DA3-LARGE-1.1] [--process_res 504] [--device cuda]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np
import torch


def write_dmb(path: Path, mat: np.ndarray) -> None:
    """Write a single-channel float32 matrix in the `WriteBinMat`/`ReadBinMat` format
    APD.cpp expects: int32 version(=1), rows, cols, cv::Mat type(=CV_32FC1=5), then raw
    row-major float32 data.
    """
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    CV_32FC1 = 5
    with open(path, "wb") as f:
        f.write(struct.pack("<iiii", 1, mat.shape[0], mat.shape[1], CV_32FC1))
        f.write(mat.tobytes())


def depth_to_disparity_255(depth: np.ndarray) -> np.ndarray:
    """DA3's raw `depth` is proportional to true distance (far = large). DVP-MVS expects
    the opposite convention on disk (near = large, like Depth-Anything's own disparity
    visualization), min-max normalized to roughly [0, 255] -- it flips it back via `255 -
    dep` before doing its own sparse-point metric rescale, so only relative structure and
    direction matter here, not absolute units.
    """
    disparity = 1.0 / np.clip(depth, 1e-6, None)
    lo, hi = np.percentile(disparity, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    normalized = np.clip((disparity - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return normalized.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense_folder",
        required=True,
        type=str,
        help="mvsnet_format dir (contains images/, cams/)",
    )
    parser.add_argument("--model", default="depth-anything/DA3-LARGE-1.1", type=str)
    parser.add_argument("--process_res", default=504, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    dense_folder = Path(args.dense_folder)
    image_dir = dense_folder / "images"
    dep_dir = dense_folder / "dep"
    dep_dir.mkdir(parents=True, exist_ok=True)

    from depth_anything_3.api import DepthAnything3

    print(f"Loading {args.model} on {args.device}...")
    model = DepthAnything3.from_pretrained(args.model).to(args.device).eval()

    image_paths = sorted(image_dir.glob("*.jpg"))
    print(f"{len(image_paths)} images in {image_dir}")

    for i, image_path in enumerate(image_paths):
        native = cv2.imread(str(image_path))
        if native is None:
            raise FileNotFoundError(f"cv2 could not read {image_path}")
        native_h, native_w = native.shape[:2]

        with torch.inference_mode():
            prediction = model.inference(
                [str(image_path)],
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )
        depth = prediction.depth[0]  # (H', W'), float, far = large

        disparity_255 = depth_to_disparity_255(depth)
        if disparity_255.shape != (native_h, native_w):
            disparity_255 = cv2.resize(
                disparity_255, (native_w, native_h), interpolation=cv2.INTER_LINEAR
            )

        out_path = dep_dir / f"{image_path.stem}.dmb"
        write_dmb(out_path, disparity_255)
        if i % 20 == 0 or i == len(image_paths) - 1:
            print(f"[{i + 1}/{len(image_paths)}] {image_path.name} -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
