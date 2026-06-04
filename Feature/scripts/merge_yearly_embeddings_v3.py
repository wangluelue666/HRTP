#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import dtype_to_descr, write_array_header_1_0

from config.path_config_v3 import PATHS
from utils.resource_manager_v3 import ResourceManagerV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge yearly V3 embeddings into sequence arrays with low memory usage."
    )
    parser.add_argument("-d", "--dataset", required=True, choices=["acm", "aps", "dblp"])
    parser.add_argument("--year_start", type=int, default=2000)
    parser.add_argument("--year_end", type=int, default=2014)
    parser.add_argument(
        "--chunk_authors",
        type=int,
        default=4096,
        help="Number of authors processed per chunk. Larger is faster but uses more memory.",
    )
    return parser.parse_args()


class NpyStreamWriter:
    def __init__(self, path: Path, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.path = Path(path)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.expected_nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        self.written_nbytes = 0

        self.f = open(self.path, "wb")
        header_dict = {
            "descr": dtype_to_descr(self.dtype),
            "fortran_order": False,
            "shape": self.shape,
        }
        write_array_header_1_0(self.f, header_dict)

    def write(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=self.dtype, order="C")
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        self.f.write(arr.tobytes(order="C"))
        self.written_nbytes += arr.nbytes

    def close(self) -> None:
        try:
            self.f.flush()
        finally:
            self.f.close()

        if self.written_nbytes != self.expected_nbytes:
            raise ValueError(
                f"File size mismatch for {self.path}: "
                f"written={self.written_nbytes}, expected={self.expected_nbytes}"
            )


def iter_author_chunks(n_authors: int, chunk_size: int):
    for start in range(0, n_authors, chunk_size):
        end = min(start + chunk_size, n_authors)
        yield start, end


def load_yearly_arrays(yearly_root: Path, prefix: str, years: list[int]) -> list[np.ndarray]:
    return [
        np.load(yearly_root / f"{prefix}_{year}.npy", mmap_mode="r")
        for year in years
    ]


def validate_shapes(
    stock_arrays: list[np.ndarray],
    flow_arrays: list[np.ndarray],
    joint_arrays: list[np.ndarray],
    mask_arrays: list[np.ndarray],
    n_authors: int,
    hidden_dim: int,
    years: list[int],
) -> None:
    for i, year in enumerate(years):
        if stock_arrays[i].shape != (n_authors, hidden_dim):
            raise ValueError(
                f"stock_{year}.npy shape mismatch: "
                f"{stock_arrays[i].shape}, expected {(n_authors, hidden_dim)}"
            )
        if flow_arrays[i].shape != (n_authors, hidden_dim):
            raise ValueError(
                f"flow_{year}.npy shape mismatch: "
                f"{flow_arrays[i].shape}, expected {(n_authors, hidden_dim)}"
            )
        if joint_arrays[i].shape != (n_authors, hidden_dim):
            raise ValueError(
                f"joint_{year}.npy shape mismatch: "
                f"{joint_arrays[i].shape}, expected {(n_authors, hidden_dim)}"
            )
        if mask_arrays[i].shape != (n_authors,):
            raise ValueError(
                f"mask_{year}.npy shape mismatch: "
                f"{mask_arrays[i].shape}, expected {(n_authors,)}"
            )


def main() -> None:
    args = parse_args()
    PATHS.ensure_v3_dirs()

    rm = ResourceManagerV3(args.dataset)
    dp = PATHS.get_dataset_paths(args.dataset)

    n_authors = len(rm.author_ids_history)
    years = list(range(args.year_start, args.year_end + 1))
    T = len(years)

    yearly_root = Path(dp.yearly_embed_root)
    out_root = Path(dp.merged_embed_root)
    out_root.mkdir(parents=True, exist_ok=True)

    first_stock = np.load(yearly_root / f"stock_{years[0]}.npy", mmap_mode="r")
    hidden_dim = first_stock.shape[1]
    del first_stock

    stock_arrays = load_yearly_arrays(yearly_root, "stock", years)
    flow_arrays = load_yearly_arrays(yearly_root, "flow", years)
    joint_arrays = load_yearly_arrays(yearly_root, "joint", years)
    mask_arrays = load_yearly_arrays(yearly_root, "mask", years)

    validate_shapes(
        stock_arrays=stock_arrays,
        flow_arrays=flow_arrays,
        joint_arrays=joint_arrays,
        mask_arrays=mask_arrays,
        n_authors=n_authors,
        hidden_dim=hidden_dim,
        years=years,
    )

    stock_writer = NpyStreamWriter(out_root / "X_stock.npy", (n_authors, T, hidden_dim), np.float32)
    flow_writer = NpyStreamWriter(out_root / "X_flow.npy", (n_authors, T, hidden_dim), np.float32)
    joint_writer = NpyStreamWriter(out_root / "X_joint.npy", (n_authors, T, hidden_dim), np.float32)
    concat_writer = NpyStreamWriter(out_root / "X_concat.npy", (n_authors, T, hidden_dim * 3), np.float32)
    mask_writer = NpyStreamWriter(out_root / "M_year.npy", (n_authors, T), np.uint8)

    try:
        for start, end in iter_author_chunks(n_authors, args.chunk_authors):
            bsz = end - start

            stock_chunk = np.empty((bsz, T, hidden_dim), dtype=np.float32)
            flow_chunk = np.empty((bsz, T, hidden_dim), dtype=np.float32)
            joint_chunk = np.empty((bsz, T, hidden_dim), dtype=np.float32)
            concat_chunk = np.empty((bsz, T, hidden_dim * 3), dtype=np.float32)
            mask_chunk = np.empty((bsz, T), dtype=np.uint8)

            for i in range(T):
                stock_slice = stock_arrays[i][start:end]
                flow_slice = flow_arrays[i][start:end]
                joint_slice = joint_arrays[i][start:end]
                mask_slice = mask_arrays[i][start:end]

                stock_chunk[:, i, :] = stock_slice
                flow_chunk[:, i, :] = flow_slice
                joint_chunk[:, i, :] = joint_slice
                mask_chunk[:, i] = mask_slice

                concat_chunk[:, i, 0:hidden_dim] = stock_slice
                concat_chunk[:, i, hidden_dim:2 * hidden_dim] = flow_slice
                concat_chunk[:, i, 2 * hidden_dim:3 * hidden_dim] = joint_slice

            stock_writer.write(stock_chunk)
            flow_writer.write(flow_chunk)
            joint_writer.write(joint_chunk)
            concat_writer.write(concat_chunk)
            mask_writer.write(mask_chunk)

    finally:
        stock_writer.close()
        flow_writer.close()
        joint_writer.close()
        concat_writer.close()
        mask_writer.close()

    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "year_start": int(args.year_start),
                "year_end": int(args.year_end),
                "n_authors": int(n_authors),
                "T": int(T),
                "hidden_dim": int(hidden_dim),
                "concat_dim": int(hidden_dim * 3),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[DONE] merged yearly embeddings")
    print(f"  X_stock : {out_root / 'X_stock.npy'}")
    print(f"  X_flow  : {out_root / 'X_flow.npy'}")
    print(f"  X_joint : {out_root / 'X_joint.npy'}")
    print(f"  X_concat: {out_root / 'X_concat.npy'}")
    print(f"  M_year  : {out_root / 'M_year.npy'}")


if __name__ == "__main__":
    main()