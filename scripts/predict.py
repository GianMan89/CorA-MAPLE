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
from cora_maple.config import PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True, help='NPZ with X, ordered tags, optional ids, lengths and sample_period_minutes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    with np.load(args.inputs, allow_pickle=False) as data:
        X = data['X'].astype(np.float32)
        tags = data['tags'].tolist()
        ids = data['ids'].tolist() if 'ids' in data else list(range(len(X)))
        lengths = np.asarray(data['lengths']) if 'lengths' in data else None
        period = float(data['sample_period_minutes']) if 'sample_period_minutes' in data else None
    info_path = args.checkpoint.with_name('model_info.json')
    if not info_path.exists():
        raise FileNotFoundError('The checkpoint requires its accompanying model_info.json label/channel map.')
    info = json.loads(info_path.read_text())
    if tags != info['channels'] or X.ndim != 3 or X.shape[1] != len(tags) or X.shape[2] < 1:
        raise ValueError('Channel identity/order or observation dimensions differ from the model configuration.')
    if len(ids) != len(X):
        raise ValueError('One source ID per observation is required.')
    if period is not None and not np.isclose(period,info['sample_period_minutes'],rtol=0,atol=1e-12):
        raise ValueError('Sampling interval differs from the trained model.')
    if lengths is not None and (lengths.shape != (len(X),) or not np.issubdtype(lengths.dtype,np.integer) or np.any(lengths<1) or np.any(lengths>X.shape[-1])):
        raise ValueError('lengths must give the positive count of real time samples for each observation.')
    if len(X) == 0 or not np.isin(X, [0, 1]).all():
        raise ValueError('Expected nonempty binary observations.')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('protocol') != PROTOCOL:
        raise ValueError('Checkpoint protocol does not match the configured clean-supervision method.')
    model = Classifier(PatchEncoder(checkpoint['n_tags'], **checkpoint['model']), checkpoint['n_classes'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(args.device).eval()
    prediction = predict_neural(model, X, args.device, lengths=lengths)
    result = pd.DataFrame({'id': ids, 'predicted_class_index': prediction,
                           'predicted_class': [info['classes'][int(i)] for i in prediction]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Wrote {len(result)} predictions to {args.output}')


if __name__ == '__main__':
    main()
