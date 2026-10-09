#!/usr/bin/env python3
"""Audit measured three-fold predictions and optionally replay clean checkpoints.

These operations read the archived models; they never fit a classifier or alter
repeated-CV workspaces. Outputs are written to a separate user-selected directory.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor")]
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from afc_excerpt_ssl.data import load_processed

ARCHIVE = ROOT / "results/reference_cv3"
STUDIES = ("tep_primary", "tep_additional_controls", "tep_absolute_scaling", "tep_long_supervised")


def audit_predictions(archive: Path) -> dict:
    recordings = load_processed(ROOT / "data/processed/tep_60min.npz")
    labels = dict(zip(recordings.ids.tolist(), recordings.y.tolist()))
    fits, rows = 0, 0
    for study in STUDIES:
        for path in sorted((archive / study).rglob("predictions.npz")):
            metrics = pd.read_csv(path.with_name("metrics.csv"))
            with np.load(path, allow_pickle=False) as saved:
                expected_y = np.asarray([labels[str(i)] for i in saved['ids']])
                if not np.array_equal(expected_y, saved['y']):
                    raise ValueError(f'Ground-truth/source mismatch: {path}')
                if len(saved['conditions']) != len(metrics):
                    raise ValueError(f'Incomplete metric conditions: {path}')
                for row_index, (condition, prediction) in enumerate(zip(saved['conditions'], saved['predictions'])):
                    operator, severity, draw = str(condition).split('|')
                    row = metrics.iloc[row_index]
                    if (row.perturbation, float(row.severity), int(row.draw)) != (operator, float(severity), int(draw)):
                        raise ValueError(f'Metric row order mismatch: {path}')
                    scores = {
                        'macro_f1': f1_score(saved['y'], prediction, labels=np.arange(len(saved['classes'])), average='macro', zero_division=0),
                        'accuracy': accuracy_score(saved['y'], prediction),
                        'balanced_accuracy': balanced_accuracy_score(saved['y'], prediction),
                    }
                    for key, value in scores.items():
                        if key in metrics and not np.isclose(value, row[key], atol=1e-12, rtol=0):
                            raise ValueError(f'{key} mismatch: {path} / {condition}')
                    rows += 1
            fits += 1
    return {'protocol': 'reference_cv3', 'prediction_files': fits, 'metric_rows_recomputed': rows,
            'metric_equality': True, 'retraining': False}


def replay_clean(archive: Path) -> dict:
    import torch
    from afc_excerpt_ssl.models import PatchEncoder, Classifier
    torch.set_num_threads(1)
    data = load_processed(ROOT / 'data/processed/tep_60min.npz')
    positions = {str(identifier): i for i, identifier in enumerate(data.ids)}
    count, instances = 0, 0
    for study in STUDIES:
        for path in sorted((archive / study).rglob('classifier.pt')):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            model = Classifier(PatchEncoder(checkpoint['n_tags'], **checkpoint['model']), checkpoint['n_classes']).eval()
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            with np.load(path.with_name('predictions.npz'), allow_pickle=False) as saved:
                indices = np.asarray([positions[str(identifier)] for identifier in saved['ids']])
                clean_rows = [i for i, key in enumerate(saved['conditions']) if str(key).split('|')[0] == 'clean']
                if len(clean_rows) != 1:
                    raise ValueError(f'Expected a single clean condition: {path}')
                predictions = []
                with torch.no_grad():
                    for start in range(0, len(indices), 128):
                        predictions.append(model(torch.from_numpy(data.X[indices[start:start + 128]])).argmax(1).numpy())
                actual = np.concatenate(predictions)
                if not np.array_equal(actual, saved['predictions'][clean_rows[0]]):
                    raise ValueError(f'Clean replay differs in the selected environment: {path}')
                instances += len(actual)
            count += 1
    return {'neural_checkpoints': count, 'clean_instance_predictions': instances, 'replay_equality': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-clean', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'validation/local/reference_results.json')
    args = parser.parse_args()
    result = audit_predictions(ARCHIVE)
    if args.replay_clean:
        result['clean_replay'] = replay_clean(ARCHIVE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
