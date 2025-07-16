from pathlib import Path

import yaml
from loguru import logger

from reflectolearn.data_processing.preprocess import preprocess_q4

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    logger.info(f"Starting preprocessing")
    config = load_config(Path("config.yml"))
    logger.info(f"Config: {config}")

    raw_file = Path(config["data"]["raw_data_dir"]) / config["data"]["file_name"]
    raw_name = Path(config["data"]["file_name"]).stem
    data_version = config["data"]["version"]
    preprocessed_file = Path(config["data"]["processed_data_dir"]) / f"{raw_name}_{data_version}.h5"

    # Preprocess the raw data
    preprocess_q4(raw_file, preprocessed_file)

if __name__ == "__main__":
    main()
