"""Download public model weights to ignored artifacts/models; record immutable revisions."""

import os, json, argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download, HfApi

p = argparse.ArgumentParser()
p.add_argument("--revision", default=None)
a = p.parse_args()
models = [
    ("bge", "BAAI/bge-small-en-v1.5"),
    ("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
]
manifest = []
for folder, repo in models:
    pinned = json.loads(
        Path("results/model-provenance.json").read_text(encoding="utf-8")
    )
    selected = a.revision or next(
        (r["revision"] for r in pinned if r["model"] == repo), "main"
    )
    revision = HfApi().model_info(repo, revision=selected).sha
    snapshot_download(
        repo,
        revision=revision,
        local_dir=Path("artifacts/models") / folder,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer*",
            "vocab*",
            "merges*",
            "*.txt",
            "1_Pooling/*",
        ],
        max_workers=2,
    )
    manifest.append({"model": repo, "revision": revision, "folder": folder})
Path("artifacts/models/manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
print(json.dumps(manifest))
