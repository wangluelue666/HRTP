#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class DatasetPaths:
    # Raw dataset json
    raw_json: Path

    # Fixed labels and split files
    label_root: Path
    split_file: Path

    # Shared author id order
    author_ids_history: Path

    # History labels
    history_mask: Path
    history_stock: Path
    history_flow: Path
    history_hs: Path
    history_hf: Path
    history_cls: Path

    # Eval labels
    eval_cls: Path

    # Meta stats
    meta_root: Path

    # Cache roots
    cache_root: Path
    cache_train_root: Path
    cache_val_root: Path
    cache_test_root: Path

    # Encoder checkpoints
    encoder_ckpt_root: Path

    # Embeddings
    yearly_embed_root: Path
    merged_embed_root: Path

    # Predictor checkpoints
    predictor_ckpt_root: Path

    # Result output
    result_root: Path


class PathConfigV3:
    """
    Unified path manager for my_method_v3.

    Cache layout (NEW)
    ------------------
    artifacts/cache/{dataset}/{split}/{year}/
      - index.npz
      - meta.json
      - shard_*.pkl

    Notes
    -----
    1. All V3 artifacts are placed under:
       /root/autodl-tmp/WH2/my_method_v3

    2. Some reusable V1 resources are linked from:
       /root/autodl-tmp/WH2/my_method/outputs/index
       /root/autodl-tmp/WH2/my_method/outputs/topic

    3. Labels and splits are fixed external inputs.
    """

    def __init__(self) -> None:
        self.root = Path("/root/autodl-tmp/WH2/my_method_v3")

        # External shared roots
        self.wh_root = Path("/root/autodl-tmp/WH2")
        self.data_root = self.wh_root / "data"
        self.v1_root = self.wh_root / "my_method"

        # V1 shared resources
        self.v1_index_root = self.v1_root / "outputs" / "index"
        self.v1_topic_root = self.v1_root / "outputs" / "topic"

        # Dataset json files
        self.raw_dataset_files: Dict[str, Path] = {
            "acm": self.data_root / "dataset" / "acm" / "acm_V11_2000_2020_v5.json",
            "aps": self.data_root / "dataset" / "aps" / "aps_2000_2020_v9.json",
            "dblp": self.data_root / "dataset" / "dblp" / "dblpV16_2000_2020_v5.json",
        }

        # Fixed labels and splits
        self.label_roots: Dict[str, Path] = {
            "acm": self.data_root / "lable" / "acm",
            "aps": self.data_root / "lable" / "aps",
            "dblp": self.data_root / "lable" / "dblp",
        }
        self.split_files: Dict[str, Path] = {
            "acm": self.data_root / "splits" / "acm" / "split_70_15_15_seed42.json",
            "aps": self.data_root / "splits" / "aps" / "split_70_15_15_seed42.json",
            "dblp": self.data_root / "splits" / "dblp" / "split_70_15_15_seed42.json",
        }

        # V3 artifact roots
        self.artifact_root = self.root / "artifacts"
        self.cache_root = self.artifact_root / "cache"
        self.encoder_ckpt_root = self.artifact_root / "encoder_ckpt"
        self.embedding_root = self.artifact_root / "embeddings"
        self.predictor_ckpt_root = self.artifact_root / "predictor_ckpt"

        # V3 output roots
        self.output_root = self.root / "outputs"
        self.log_root = self.output_root / "logs"
        self.result_root = self.output_root / "results"
        self.analysis_root = self.output_root / "analysis"

        # Config/script/model roots
        self.config_root = self.root / "config"
        self.script_root = self.root / "scripts"
        self.dataset_root = self.root / "dataset"
        self.subgraph_root = self.root / "subgraph"
        self.model_root = self.root / "models"
        self.encoder_model_root = self.model_root / "encoder"
        self.predictor_model_root = self.model_root / "predictor"
        self.util_root = self.root / "utils"

    # ------------------------------------------------------------------
    # Core dataset-level paths
    # ------------------------------------------------------------------
    def get_dataset_paths(self, dataset: str) -> DatasetPaths:
        dataset = dataset.lower().strip()
        self._validate_dataset(dataset)

        label_root = self.label_roots[dataset]
        history_root = label_root / "history"
        eval_root = label_root / "eval"
        meta_root = label_root / "meta"

        cache_root = self.cache_root / dataset
        yearly_embed_root = self.embedding_root / dataset / "yearly"
        merged_embed_root = self.embedding_root / dataset / "merged"

        return DatasetPaths(
            raw_json=self.raw_dataset_files[dataset],
            label_root=label_root,
            split_file=self.split_files[dataset],
            author_ids_history=history_root / "author_ids_active_until_2014.json",
            history_mask=history_root / "mask_2000_2014.npy",
            history_stock=history_root / "stock_2000_2014.npy",
            history_flow=history_root / "flow_2000_2014.npy",
            history_hs=history_root / "hs_2000_2014.npy",
            history_hf=history_root / "hf_2000_2014.npy",
            history_cls=history_root / "cls_2000_2014.npy",
            eval_cls=eval_root / "cls_2015_2020.npy",
            meta_root=meta_root,
            cache_root=cache_root,
            cache_train_root=cache_root / "train",
            cache_val_root=cache_root / "val",
            cache_test_root=cache_root / "test",
            encoder_ckpt_root=self.encoder_ckpt_root / dataset,
            yearly_embed_root=yearly_embed_root,
            merged_embed_root=merged_embed_root,
            predictor_ckpt_root=self.predictor_ckpt_root / dataset,
            result_root=self.result_root / dataset,
        )

    # ------------------------------------------------------------------
    # NEW: year-split cache helpers
    # ------------------------------------------------------------------
    def get_cache_split_root(self, dataset: str, split: str) -> Path:
        dataset = dataset.lower().strip()
        split = split.lower().strip()
        self._validate_dataset(dataset)
        self._validate_split(split)

        dp = self.get_dataset_paths(dataset)
        if split == "train":
            return dp.cache_train_root
        if split == "val":
            return dp.cache_val_root
        return dp.cache_test_root

    def get_cache_year_root(self, dataset: str, split: str, year: int) -> Path:
        self._validate_year(year)
        return self.get_cache_split_root(dataset, split) / str(int(year))

    # ------------------------------------------------------------------
    # Directory creation
    # ------------------------------------------------------------------
    def ensure_v3_dirs(self) -> None:
        dirs = [
            self.root,
            self.config_root,
            self.script_root,
            self.dataset_root,
            self.subgraph_root,
            self.model_root,
            self.encoder_model_root,
            self.predictor_model_root,
            self.util_root,
            self.artifact_root,
            self.cache_root,
            self.encoder_ckpt_root,
            self.embedding_root,
            self.predictor_ckpt_root,
            self.output_root,
            self.log_root,
            self.result_root,
            self.analysis_root,
        ]
        for p in dirs:
            p.mkdir(parents=True, exist_ok=True)

        for dataset in ("acm", "aps", "dblp"):
            dp = self.get_dataset_paths(dataset)

            # split roots
            for p in [
                dp.cache_root,
                dp.cache_train_root,
                dp.cache_val_root,
                dp.cache_test_root,
                dp.encoder_ckpt_root,
                dp.yearly_embed_root,
                dp.merged_embed_root,
                dp.predictor_ckpt_root,
                dp.result_root,
            ]:
                p.mkdir(parents=True, exist_ok=True)

            # NEW: year roots under each split
            for split in ("train", "val", "test"):
                for year in range(2000, 2015):
                    self.get_cache_year_root(dataset, split, year).mkdir(parents=True, exist_ok=True)

        for p in [
            self.log_root / "cache",
            self.log_root / "encoder",
            self.log_root / "embedding",
            self.log_root / "predictor",
            self.log_root / "eval",
            self.analysis_root / "cache_stats",
            self.analysis_root / "feature_stats",
            self.analysis_root / "debug_samples",
        ]:
            p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_required_inputs(self, dataset: str) -> None:
        dp = self.get_dataset_paths(dataset)
        required_paths = [
            dp.raw_json,
            dp.split_file,
            dp.author_ids_history,
            dp.history_mask,
            dp.history_stock,
            dp.history_flow,
            dp.history_hs,
            dp.history_hf,
            dp.history_cls,
            dp.eval_cls,
            self.v1_index_root,
            self.v1_topic_root,
        ]
        missing = [str(p) for p in required_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required input paths for dataset "
                f"{dataset}:\n" + "\n".join(missing)
            )

    def _validate_dataset(self, dataset: str) -> None:
        if dataset not in {"acm", "aps", "dblp"}:
            raise ValueError(f"Unsupported dataset: {dataset}")

    @staticmethod
    def _validate_split(split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

    @staticmethod
    def _validate_year(year: int) -> None:
        if int(year) < 2000 or int(year) > 2014:
            raise ValueError(f"Unsupported year for history cache: {year}")


PATHS = PathConfigV3()


if __name__ == "__main__":
    PATHS.ensure_v3_dirs()
    for ds in ("acm", "aps", "dblp"):
        dp = PATHS.get_dataset_paths(ds)
        print(f"[{ds}]")
        print(f"  raw_json         : {dp.raw_json}")
        print(f"  split_file       : {dp.split_file}")
        print(f"  cache_root       : {dp.cache_root}")
        print(f"  train_2000_root  : {PATHS.get_cache_year_root(ds, 'train', 2000)}")
        print(f"  encoder_ckpt_root: {dp.encoder_ckpt_root}")
        print(f"  yearly_embed_root: {dp.yearly_embed_root}")
        print(f"  merged_embed_root: {dp.merged_embed_root}")
        print(f"  result_root      : {dp.result_root}")