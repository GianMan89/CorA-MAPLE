"""Fixed published-reference configurations; no tuning on held-out data."""
from afc_robustness.models.eac import EAC1NN
from afc_robustness.models.mbw_lr import MBWLogisticRegression
from afc_robustness.models.jaccard import Jaccard1NN
from afc_robustness.models.wdi import WDI1NN
from afc_robustness.models.acm_svm import ACMSVM


def make_baseline(name: str, settings: dict, seed: int):
    classes = {"EAC-1NN": EAC1NN, "MBW-LR": MBWLogisticRegression,
               "JAC-1NN": Jaccard1NN, "WDI-1NN": WDI1NN, "ACM-SVM": ACMSVM}
    if name not in classes:
        raise ValueError(f"Unknown baseline {name}")
    args = dict(settings[name])
    if name == "ACM-SVM":
        args["random_state"] = seed
        args["probability"] = False
    return classes[name](**args)


def predict_baseline(model, X):
    if isinstance(model, ACMSVM):
        # Retain native SVC decisions; do not introduce Platt calibration at k=1.
        return model.clf_.predict(model.coactivation_features(X))
    return model.predict(X)
