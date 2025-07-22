from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict[str, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
