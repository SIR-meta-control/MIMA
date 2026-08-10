import hashlib
import json
from pathlib import Path


def test_packaged_baseline_weights_match_recorded_hashes():
    model_dir = Path(__file__).resolve().parents[1] / "models" / "baselines"
    manifest = json.loads((model_dir / "checksums.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        digest = hashlib.sha256((model_dir / artifact["path"]).read_bytes()).hexdigest()
        assert digest == artifact["sha256"]
