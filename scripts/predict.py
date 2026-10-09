#!/usr/bin/env python3
"""Classify binary alarm observations with a saved neural checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from afc_excerpt_ssl.models import PatchEncoder, Classifier
from cora_maple.learning import predict_neural


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True, help='NPZ with X, tags and optional ids arrays')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    with np.load(args.inputs, allow_pickle=False) as data:
        X = data['X'].astype(np.float32)
        tags = data['tags'].tolist()
        ids = data['ids'].tolist() if 'ids' in data else list(range(len(X)))
    info_path = args.checkpoint.with_name('model_info.json')
    if not info_path.exists():
        raise FileNotFoundError('The checkpoint requires its accompanying model_info.json label/channel map.')
    info = json.loads(info_path.read_text())
    if tags != info['channels'] or X.ndim != 3 or X.shape[1:] != (len(tags), info['observation_samples']):
        raise ValueError('Channel identity/order or observation dimensions differ from the model configuration.')
    if len(X) == 0 or not np.isin(X, [0, 1]).all():
        raise ValueError('Expected nonempty binary observations.')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = Classifier(PatchEncoder(checkpoint['n_tags'], **checkpoint['model']), checkpoint['n_classes'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(args.device).eval()
    prediction = predict_neural(model, X, args.device)
    result = pd.DataFrame({'id': ids, 'predicted_class_index': prediction,
                           'predicted_class': [info['classes'][int(i)] for i in prediction]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Wrote {len(result)} predictions to {args.output}')


if __name__ == '__main__':
    main()
