#!/usr/bin/env python3
"""Replay completed neural classifiers against their saved paired test inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from afc_excerpt_ssl.models import PatchEncoder, Classifier
from cora_maple.common import atomic_json, file_hash
from cora_maple.learning import predict_neural
from cora_maple.report import load_study
from cora_maple.runner import completed, task_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('runs/tep_cv10x5_cleanft_v2_1'))
    parser.add_argument('--study', default='primary')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    plan, cfg = load_study(workspace, args.study)
    if any(not completed(workspace, task) for task in plan['tasks'].values()):
        raise RuntimeError('Replay requires a complete study.')
    torch.set_num_threads(1)
    checked = conditions = 0
    for ident in plan['evaluations']:
        task = plan['tasks'][ident]
        if task['kind'] != 'fit':
            continue
        directory = task_directory(workspace, ident)
        checkpoint = torch.load(directory / 'classifier.pt', map_location='cpu', weights_only=True)
        model = Classifier(PatchEncoder(checkpoint['n_tags'], **checkpoint['model']), checkpoint['n_classes'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.to(args.device).eval()
        evaluation = json.loads((directory / 'evaluation.json').read_text())
        cache_path = workspace / evaluation['test_cache']
        if file_hash(cache_path) != evaluation['test_cache_sha256']:
            raise ValueError(f'Changed test cache: {cache_path}')
        with np.load(cache_path, allow_pickle=False) as cache, np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
            for key in ('conditions', 'ids', 'y', 'classes', 'lengths'):
                if not np.array_equal(cache[key], saved[key]):
                    raise ValueError(f'Prediction/cache identity mismatch: {ident} / {key}')
            for index, observation in enumerate(cache['X']):
                actual = predict_neural(model, observation, args.device, cfg['test']['inference_batch'], cache["lengths"][index])
                if not np.array_equal(actual, saved['predictions'][index]):
                    raise ValueError(f'Replay differs: {ident} / {cache["conditions"][index]}')
                conditions += 1
        checked += 1
    result = {'study': args.study, 'neural_models': checked, 'model_conditions': conditions,
              'prediction_equality': True, 'device': args.device}
    atomic_json(workspace / 'reports' / args.study / 'checkpoint_replay.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
