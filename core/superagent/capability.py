"""Offline forward probe in the solution interpreter, with pinned weight identity."""

import hashlib
import json
import sys
from pathlib import Path


def verify(hub_id, snapshot, library):
    import torch

    root = Path(snapshot)
    weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.bin"))
    if not weights:
        raise ValueError(f"no model weights in {root}")
    hashes = {}
    for path in weights:
        with path.open("rb") as f:
            hashes[path.name] = hashlib.file_digest(f, "sha256").hexdigest()
    with torch.inference_mode():
        if library == "timm":
            import timm

            name = hub_id.split("/")[-1]
            kw = {"img_size": 224} if name.startswith("vit_") else {}
            net = timm.create_model(
                name,
                pretrained=True,
                num_classes=0,
                pretrained_cfg_overlay={"file": str(weights[0])},
                **kw,
            )
            output = net.eval()(torch.zeros(1, 3, 224, 224))
        elif library == "embedding":
            from sentence_transformers import SentenceTransformer

            output = SentenceTransformer(str(root), device="cpu").encode(
                ["check"], convert_to_tensor=True
            )
        else:
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
            output = (
                AutoModel.from_pretrained(root, local_files_only=True)
                .eval()(**tok("check", return_tensors="pt"))
                .last_hidden_state
            )
    if not output.numel() or not torch.isfinite(output).all():
        raise ValueError("non-finite or empty forward output")
    return dict(
        state="forward_ok",
        runner=sys.executable,
        snapshot=str(root),
        revision=root.name,
        weights=hashes,
        shape=list(output.shape),
    )


if __name__ == "__main__":
    print(json.dumps(verify(*sys.argv[1:])))
