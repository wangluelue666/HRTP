#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict


VALID_DATASETS = {"acm", "aps", "dblp"}
DEFAULT_MODEL_NAME = "HARP"


@dataclass(frozen=True)
class HARPPathConfig:
    # Basic identifiers
    dataset: str
    model_name: str

    # Project roots
    project_root: str
    source_feature_root: str
    data_root: str
    result_root: str

    # External read-only inputs
    merged_root: str
    x_stock_path: str
    x_flow_path: str
    x_joint_path: str
    m_year_path: str

    label_root: str
    cls_label_path: str
    split_path: str
    author_ids_path: str
    meta_root: str

    # Project internal roots
    config_root: str
    dataset_root: str
    losses_root: str
    models_root: str
    scripts_root: str
    artifacts_root: str
    outputs_root: str

    cache_root: str
    processed_root: str
    predictor_ckpt_root: str
    analysis_root: str
    logs_root: str

    cache_stats_root: str
    cache_debug_samples_root: str
    cache_analysis_root: str

    processed_predictor_inputs_root: str
    processed_temporal_meta_root: str

    train_log_root: str

    output_train_root: str
    output_predict_root: str
    output_reports_root: str

    final_result_model_root: str
    final_pred_path: str
    final_author_ids_out_path: str

    # Common internal files
    input_manifest_path: str
    split_summary_path: str
    class_stats_train_path: str
    config_snapshot_path: str

    train_indices_path: str
    val_indices_path: str
    test_indices_path: str
    valid_len_path: str
    year_ids_path: str
    recency_ids_path: str

    predictor_best_ckpt_path: str
    predictor_last_ckpt_path: str
    predictor_train_state_path: str

    attention_maps_root: str
    decoder_states_root: str
    analysis_class_stats_root: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _validate_dataset(dataset: str) -> str:
    dataset = dataset.lower().strip()
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Expected one of: {sorted(VALID_DATASETS)}"
        )
    return dataset


def build_harp_path_config(
    dataset: str,
    model_name: str = DEFAULT_MODEL_NAME,
    project_root: str = "/root/autodl-tmp/WH2/HARP",
    source_feature_root: str = "/root/autodl-tmp/WH2/my_method_v3",
    data_root: str = "/root/autodl-tmp/WH2/data",
    result_root: str = "/root/autodl-tmp/WH2/results",
) -> HARPPathConfig:
    """
    Build all project paths for HARP.

    External read-only inputs:
        - merged sequence features from my_method_v3
        - labels / split / author ids from the shared data root
    """
    dataset = _validate_dataset(dataset)
    model_name = model_name.strip()

    project_root_p = Path(project_root)
    source_feature_root_p = Path(source_feature_root)
    data_root_p = Path(data_root)
    result_root_p = Path(result_root)

    # ------------------------------------------------------------------
    # External read-only inputs
    # ------------------------------------------------------------------
    merged_root = source_feature_root_p / "artifacts" / "embeddings" / dataset / "merged"
    x_stock_path = merged_root / "X_stock.npy"
    x_flow_path = merged_root / "X_flow.npy"
    x_joint_path = merged_root / "X_joint.npy"
    m_year_path = merged_root / "M_year.npy"

    # Note: the real folder name is "lable", not "label"
    label_root = data_root_p / "lable" / dataset
    cls_label_path = label_root / "eval" / "cls_2015_2020.npy"
    split_path = data_root_p / "splits" / dataset / "split_70_15_15_seed42.json"
    author_ids_path = data_root_p / "active_list" / f"author_ids_active_until_2014_{dataset}.json"
    meta_root = label_root / "meta"

    # ------------------------------------------------------------------
    # Internal project roots
    # ------------------------------------------------------------------
    config_root = project_root_p / "config"
    dataset_root = project_root_p / "dataset"
    losses_root = project_root_p / "losses"
    models_root = project_root_p / "models"
    scripts_root = project_root_p / "scripts"
    artifacts_root = project_root_p / "artifacts"
    outputs_root = project_root_p / "outputs"

    cache_root = artifacts_root / "cache" / dataset
    processed_root = artifacts_root / "processed" / dataset
    predictor_ckpt_root = artifacts_root / "predictor_ckpt" / dataset / model_name
    analysis_root = artifacts_root / "analysis" / dataset
    logs_root = artifacts_root / "logs" / dataset

    cache_stats_root = cache_root / "stats"
    cache_debug_samples_root = cache_root / "debug_samples"
    cache_analysis_root = cache_root / "analysis"

    processed_predictor_inputs_root = processed_root / "predictor_inputs"
    processed_temporal_meta_root = processed_root / "temporal_meta"

    train_log_root = logs_root / "train"

    output_train_root = outputs_root / dataset / "train"
    output_predict_root = outputs_root / dataset / "predict"
    output_reports_root = outputs_root / dataset / "reports"

    final_result_model_root = result_root_p / model_name
    final_pred_path = final_result_model_root / f"pred_{model_name}_cls_2015_2020_{dataset}.npy"
    final_author_ids_out_path = final_result_model_root / f"author_ids_{dataset}.json"

    # ------------------------------------------------------------------
    # Common internal files
    # ------------------------------------------------------------------
    input_manifest_path = cache_stats_root / "input_manifest.json"
    split_summary_path = cache_stats_root / "split_summary.json"
    class_stats_train_path = cache_stats_root / "class_stats_train.json"
    config_snapshot_path = predictor_ckpt_root / "config_snapshot.json"

    train_indices_path = processed_predictor_inputs_root / "train_indices.npy"
    val_indices_path = processed_predictor_inputs_root / "val_indices.npy"
    test_indices_path = processed_predictor_inputs_root / "test_indices.npy"
    valid_len_path = processed_predictor_inputs_root / "valid_len.npy"
    year_ids_path = processed_predictor_inputs_root / "year_ids.npy"
    recency_ids_path = processed_predictor_inputs_root / "recency_ids.npy"

    predictor_best_ckpt_path = predictor_ckpt_root / "best.pt"
    predictor_last_ckpt_path = predictor_ckpt_root / "last.pt"
    predictor_train_state_path = predictor_ckpt_root / "train_state.json"

    attention_maps_root = analysis_root / "attention_maps"
    decoder_states_root = analysis_root / "decoder_states"
    analysis_class_stats_root = analysis_root / "class_stats"

    return HARPPathConfig(
        dataset=dataset,
        model_name=model_name,

        project_root=str(project_root_p),
        source_feature_root=str(source_feature_root_p),
        data_root=str(data_root_p),
        result_root=str(result_root_p),

        merged_root=str(merged_root),
        x_stock_path=str(x_stock_path),
        x_flow_path=str(x_flow_path),
        x_joint_path=str(x_joint_path),
        m_year_path=str(m_year_path),

        label_root=str(label_root),
        cls_label_path=str(cls_label_path),
        split_path=str(split_path),
        author_ids_path=str(author_ids_path),
        meta_root=str(meta_root),

        config_root=str(config_root),
        dataset_root=str(dataset_root),
        losses_root=str(losses_root),
        models_root=str(models_root),
        scripts_root=str(scripts_root),
        artifacts_root=str(artifacts_root),
        outputs_root=str(outputs_root),

        cache_root=str(cache_root),
        processed_root=str(processed_root),
        predictor_ckpt_root=str(predictor_ckpt_root),
        analysis_root=str(analysis_root),
        logs_root=str(logs_root),

        cache_stats_root=str(cache_stats_root),
        cache_debug_samples_root=str(cache_debug_samples_root),
        cache_analysis_root=str(cache_analysis_root),

        processed_predictor_inputs_root=str(processed_predictor_inputs_root),
        processed_temporal_meta_root=str(processed_temporal_meta_root),

        train_log_root=str(train_log_root),

        output_train_root=str(output_train_root),
        output_predict_root=str(output_predict_root),
        output_reports_root=str(output_reports_root),

        final_result_model_root=str(final_result_model_root),
        final_pred_path=str(final_pred_path),
        final_author_ids_out_path=str(final_author_ids_out_path),

        input_manifest_path=str(input_manifest_path),
        split_summary_path=str(split_summary_path),
        class_stats_train_path=str(class_stats_train_path),
        config_snapshot_path=str(config_snapshot_path),

        train_indices_path=str(train_indices_path),
        val_indices_path=str(val_indices_path),
        test_indices_path=str(test_indices_path),
        valid_len_path=str(valid_len_path),
        year_ids_path=str(year_ids_path),
        recency_ids_path=str(recency_ids_path),

        predictor_best_ckpt_path=str(predictor_best_ckpt_path),
        predictor_last_ckpt_path=str(predictor_last_ckpt_path),
        predictor_train_state_path=str(predictor_train_state_path),

        attention_maps_root=str(attention_maps_root),
        decoder_states_root=str(decoder_states_root),
        analysis_class_stats_root=str(analysis_class_stats_root),
    )


def get_required_dirs(cfg: HARPPathConfig) -> Dict[str, str]:
    """
    Return all directories that should be created before running the project.
    """
    return {
        "project_root": cfg.project_root,
        "config_root": cfg.config_root,
        "dataset_root": cfg.dataset_root,
        "losses_root": cfg.losses_root,
        "models_root": cfg.models_root,
        "scripts_root": cfg.scripts_root,
        "artifacts_root": cfg.artifacts_root,
        "outputs_root": cfg.outputs_root,

        "cache_root": cfg.cache_root,
        "processed_root": cfg.processed_root,
        "predictor_ckpt_root": cfg.predictor_ckpt_root,
        "analysis_root": cfg.analysis_root,
        "logs_root": cfg.logs_root,

        "cache_stats_root": cfg.cache_stats_root,
        "cache_debug_samples_root": cfg.cache_debug_samples_root,
        "cache_analysis_root": cfg.cache_analysis_root,

        "processed_predictor_inputs_root": cfg.processed_predictor_inputs_root,
        "processed_temporal_meta_root": cfg.processed_temporal_meta_root,

        "train_log_root": cfg.train_log_root,

        "output_train_root": cfg.output_train_root,
        "output_predict_root": cfg.output_predict_root,
        "output_reports_root": cfg.output_reports_root,

        "final_result_model_root": cfg.final_result_model_root,

        "attention_maps_root": cfg.attention_maps_root,
        "decoder_states_root": cfg.decoder_states_root,
        "analysis_class_stats_root": cfg.analysis_class_stats_root,
    }


def ensure_required_dirs(cfg: HARPPathConfig) -> None:
    """
    Create all internal directories required by HARP.
    """
    for _, path_str in get_required_dirs(cfg).items():
        Path(path_str).mkdir(parents=True, exist_ok=True)


def dump_path_manifest(cfg: HARPPathConfig, save_path: str | None = None) -> None:
    """
    Dump the full path configuration to a json manifest.
    """
    if save_path is None:
        save_path = cfg.input_manifest_path

    save_path_p = Path(save_path)
    save_path_p.parent.mkdir(parents=True, exist_ok=True)

    with save_path_p.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)


def check_required_input_files(cfg: HARPPathConfig) -> Dict[str, bool]:
    """
    Check the existence of all required external read-only inputs.
    """
    return {
        "x_stock_path": Path(cfg.x_stock_path).is_file(),
        "x_flow_path": Path(cfg.x_flow_path).is_file(),
        "x_joint_path": Path(cfg.x_joint_path).is_file(),
        "m_year_path": Path(cfg.m_year_path).is_file(),
        "cls_label_path": Path(cfg.cls_label_path).is_file(),
        "split_path": Path(cfg.split_path).is_file(),
        "author_ids_path": Path(cfg.author_ids_path).is_file(),
    }


def assert_required_input_files(cfg: HARPPathConfig) -> None:
    """
    Raise FileNotFoundError if any required external input is missing.
    """
    checks = check_required_input_files(cfg)
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        details = "\n".join([f"- {name}: {getattr(cfg, name)}" for name in missing])
        raise FileNotFoundError(
            "Missing required input files:\n"
            f"{details}"
        )


if __name__ == "__main__":
    cfg = build_harp_path_config(dataset="aps", model_name=DEFAULT_MODEL_NAME)
    ensure_required_dirs(cfg)
    dump_path_manifest(cfg)
    checks = check_required_input_files(cfg)

    print("[INFO] Built HARP path config successfully.")
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
    print("[INFO] Input file existence check:")
    print(json.dumps(checks, ensure_ascii=False, indent=2))