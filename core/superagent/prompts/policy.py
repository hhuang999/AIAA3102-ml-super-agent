"""Pretrained-model policy texts (one per mode)."""

from __future__ import annotations

USAGE_ADVICE = """Advice on pretrained models:
- With little data, or data far from the pretraining domain: freeze the backbone, extract features, and train only a small head or a classical model on them.
- With more data: fine-tune the whole model.
- Mixed precision speeds up fine-tuning; follow the GPU habits in the implementation rules.
- The most common cause of timeouts is slow data loading. Use multi-process data loading; when the backbone is heavy, extract features once and cache them in ./work.
- Pretrained models are optional. For tabular data, gradient boosted trees are usually the first thing to try."""

_CATALOG = """## Pretrained-model policy
The pretrained models listed in the catalog below are stored locally and can be loaded offline with exactly the model ids shown. There is no network access: any other model name cannot be loaded. Prefer a listed pretrained model when it suits the task.
""" + USAGE_ADVICE

_OPEN = """## Pretrained-model policy
The catalog below lists recommended methods and pretrained models. Network access is available, so other public models may also be used. Downloads count against the script's time limit, and gated models that need an access request are not available.
""" + USAGE_ADVICE

_OFF = """## Pretrained-model policy
Pretrained weights are FORBIDDEN. Do not download anything from a model hub, do not load any external weight file, and do not use any pretrained model or pretrained embedding. Every model must be trained from scratch using only this task's data. There is no network access."""


def policy_paragraph(mode: str) -> str:
    return {"catalog": _CATALOG, "open": _OPEN, "off": _OFF}[mode]


RESOURCE_LINE = {
    "catalog": "Pretrained models listed in the method catalog can be loaded offline from the local cache; there is no network access.",
    "open": "Network access is available: public pretrained models may be downloaded (download time counts against the time limit).",
    "off": "There is no network access, and pretrained weights of any kind must not be used.",
}

DATA_PREP_PRETRAINED_HINT = (
    "Later steps may use a pretrained model. Prepare the data with its input requirements in mind "
    "(images: size, normalisation and format; text: tokenisation-friendly raw strings and format)."
)

MODEL_DEF_PRETRAINED_HINT = (
    "Prefer a suitable pretrained model from the method catalog. Design your own architecture only "
    "when the catalog models are clearly unsuitable for this task."
)

MODEL_DEF_OFF_HINT = "The model must be designed and trained from scratch; pretrained weights are forbidden."

REVIEW_TRUST_CATALOG = (
    "Models listed in the method catalog are available and loadable, including model names you do not recognise. "
    "Never replace or downgrade the chosen model, never raise issues because a library version might be incompatible "
    "or a model name is unknown to you, and never change constants that define model names, backbones or weight ids."
)

REVIEW_OFF = (
    "Pretrained weights are forbidden in this run. Any use of pretrained weights (from_pretrained, pretrained=True, "
    "hub downloads, pretrained embeddings, foundation models) is a problem that MUST be fixed by switching to a model "
    "trained from scratch."
)
