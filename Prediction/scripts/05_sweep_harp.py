#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


VALID_DATASETS = {"acm", "aps", "dblp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed 24-trial HARP hyperparameter sweep."
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=sorted(VALID_DATASETS))
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--project_root", type=str, default="/root/autodl-tmp/WH2/HARP")
    parser.add_argument("--source_feature_root", type=str, default="/root/autodl-tmp/WH2/my_method_v3")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/WH2/data")
    parser.add_argument("--result_root", type=str, default="/root/autodl-tmp/WH2/results/HARP_HYPER")
    return parser.parse_args()


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_candidate_table() -> List[Dict[str, Any]]:
    rows = [
        ("trial_0001", 256, 512, 2, 0.1, 0.2, 1e-3),
        ("trial_0002", 128, 512, 2, 0.1, 0.2, 1e-3),
        ("trial_0003", 384, 512, 2, 0.1, 0.2, 1e-3),
        ("trial_0004", 256, 768, 2, 0.1, 0.2, 1e-3),
        ("trial_0005", 256, 1024, 2, 0.1, 0.2, 1e-3),
        ("trial_0006", 256, 512, 1, 0.1, 0.2, 1e-3),
        ("trial_0007", 256, 512, 3, 0.1, 0.2, 1e-3),
        ("trial_0008", 256, 512, 2, 0.0, 0.2, 1e-3),
        ("trial_0009", 256, 512, 2, 0.2, 0.2, 1e-3),
        ("trial_0010", 256, 512, 2, 0.1, 0.0, 1e-3),
        ("trial_0011", 256, 512, 2, 0.1, 0.1, 1e-3),
        ("trial_0012", 256, 512, 2, 0.1, 0.4, 1e-3),
        ("trial_0013", 256, 512, 2, 0.1, 0.2, 3e-4),
        ("trial_0014", 256, 512, 2, 0.1, 0.2, 5e-4),
        ("trial_0015", 384, 1024, 3, 0.1, 0.2, 1e-3),
        ("trial_0016", 384, 1024, 2, 0.2, 0.2, 5e-4),
        ("trial_0017", 384, 768, 2, 0.1, 0.1, 5e-4),
        ("trial_0018", 384, 768, 3, 0.2, 0.1, 5e-4),
        ("trial_0019", 128, 512, 1, 0.0, 0.2, 1e-3),
        ("trial_0020", 128, 512, 2, 0.2, 0.4, 5e-4),
        ("trial_0021", 256, 768, 3, 0.1, 0.1, 5e-4),
        ("trial_0022", 256, 1024, 2, 0.2, 0.0, 5e-4),
        ("trial_0023", 384, 1024, 2, 0.0, 0.0, 3e-4),
        ("trial_0024", 128, 768, 3, 0.2, 0.4, 3e-4),
    ]

    out = []
    for trial_id, d_model, ffn_dim, encoder_layers, dropout, ss_end, lr in rows:
        out.append(
            {
                "trial_id": trial_id,
                "d_model": d_model,
                "ffn_dim": ffn_dim,
                "encoder_layers": encoder_layers,
                "dropout": dropout,
                "scheduled_sampling_end": ss_end,
                "lr": lr,
            }
        )
    return out


def is_trial_finished(trial_root: Path) -> bool:
    result_path = trial_root / "trial_result.json"
    if not result_path.is_file():
        return False
    try:
        result = load_json(result_path)
    except Exception:
        return False

    required_keys = {"trial_id", "dataset", "best_epoch", "best_val_macro_f1", "train", "val", "test", "config"}
    return required_keys.issubset(set(result.keys()))


def build_summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trial_id": result["trial_id"],
        "best_epoch": result["best_epoch"],
        "best_val_macro_f1": result["best_val_macro_f1"],
        "test_macro_f1": result["test"]["macro_f1"],
        "test_balanced_acc": result["test"]["balanced_acc"],
        "year_2015_macro_f1": result["test"]["year_2015_macro_f1"],
        "year_2016_macro_f1": result["test"]["year_2016_macro_f1"],
        "year_2017_macro_f1": result["test"]["year_2017_macro_f1"],
        "year_2018_macro_f1": result["test"]["year_2018_macro_f1"],
        "year_2019_macro_f1": result["test"]["year_2019_macro_f1"],
        "year_2020_macro_f1": result["test"]["year_2020_macro_f1"],
        "d_model": result["config"]["d_model"],
        "ffn_dim": result["config"]["ffn_dim"],
        "encoder_layers": result["config"]["encoder_layers"],
        "dropout": result["config"]["dropout"],
        "scheduled_sampling_end": result["config"]["scheduled_sampling_end"],
        "lr": result["config"]["lr"],
    }


def write_summary_files(dataset_root: Path, summary_rows: List[Dict[str, Any]]) -> None:
    summary_rows = sorted(
        summary_rows,
        key=lambda x: (-x["best_val_macro_f1"], -x["test_macro_f1"], x["trial_id"])
    )

    save_json(summary_rows, dataset_root / "summary.json")

    if summary_rows:
        with (dataset_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        save_json(summary_rows[0], dataset_root / "best_trial.json")


def main() -> None:
    args = parse_args()

    script_root = Path(args.project_root) / "scripts"
    train_script = script_root / "04_Hyperparameter_train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"Missing script: {train_script}")

    dataset_root = Path(args.result_root) / args.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)

    candidates = get_candidate_table()
    save_json(candidates, dataset_root / "candidate_table.json")

    failed_trials_path = dataset_root / "failed_trials.json"
    failed_trials: List[Dict[str, Any]] = []
    if failed_trials_path.is_file():
        try:
            failed_trials = load_json(failed_trials_path)
        except Exception:
            failed_trials = []

    summary_rows: List[Dict[str, Any]] = []
    summary_json_path = dataset_root / "summary.json"
    if summary_json_path.is_file():
        try:
            summary_rows = load_json(summary_json_path)
        except Exception:
            summary_rows = []

    done_trial_ids = {row["trial_id"] for row in summary_rows if "trial_id" in row}

    for trial in candidates:
        trial_id = trial["trial_id"]
        trial_root = dataset_root / trial_id
        trial_root.mkdir(parents=True, exist_ok=True)

        if trial_id in done_trial_ids and is_trial_finished(trial_root):
            print(f"[SKIP] {trial_id} already finished.")
            continue

        if is_trial_finished(trial_root):
            result = load_json(trial_root / "trial_result.json")
            row = build_summary_row(result)
            summary_rows = [r for r in summary_rows if r.get("trial_id") != trial_id]
            summary_rows.append(row)
            write_summary_files(dataset_root, summary_rows)
            print(f"[SYNC] {trial_id} summary restored from existing trial_result.json")
            continue

        stdout_log = trial_root / "stdout.log"
        stderr_log = trial_root / "stderr.log"

        cmd = [
            sys.executable,
            str(train_script),
            "-d", args.dataset,
            "--device", args.device,
            "--trial_id", trial_id,
            "--trial_root", str(trial_root),
            "--project_root", args.project_root,
            "--source_feature_root", args.source_feature_root,
            "--data_root", args.data_root,
            "--d_model", str(trial["d_model"]),
            "--ffn_dim", str(trial["ffn_dim"]),
            "--encoder_layers", str(trial["encoder_layers"]),
            "--dropout", str(trial["dropout"]),
            "--scheduled_sampling_end", str(trial["scheduled_sampling_end"]),
            "--lr", str(trial["lr"]),
        ]

        print("=" * 100)
        print(f"[SWEEP] Launch {trial_id} on dataset={args.dataset}, device={args.device}")
        print("[CMD]", " ".join(cmd))

        with stdout_log.open("w", encoding="utf-8") as fout, stderr_log.open("w", encoding="utf-8") as ferr:
            proc = subprocess.run(cmd, stdout=fout, stderr=ferr, check=False)

        if proc.returncode != 0:
            print(f"[FAIL] {trial_id} failed with return code {proc.returncode}")
            failed_trials.append(
                {
                    "trial_id": trial_id,
                    "returncode": proc.returncode,
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                }
            )
            save_json(failed_trials, failed_trials_path)
            continue

        result_path = trial_root / "trial_result.json"
        if not result_path.is_file():
            print(f"[FAIL] {trial_id} finished without trial_result.json")
            failed_trials.append(
                {
                    "trial_id": trial_id,
                    "returncode": -999,
                    "reason": "missing trial_result.json",
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                }
            )
            save_json(failed_trials, failed_trials_path)
            continue

        result = load_json(result_path)
        row = build_summary_row(result)

        summary_rows = [r for r in summary_rows if r.get("trial_id") != trial_id]
        summary_rows.append(row)
        write_summary_files(dataset_root, summary_rows)

    print("=" * 100)
    print("[DONE] Sweep finished.")
    print(f"[DONE] dataset={args.dataset}")
    print(f"[DONE] summary={dataset_root / 'summary.json'}")
    print(f"[DONE] best={dataset_root / 'best_trial.json'}")
    print(f"[DONE] failed={failed_trials_path}")


if __name__ == "__main__":
    main()