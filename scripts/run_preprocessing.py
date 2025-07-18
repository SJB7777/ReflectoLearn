import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from loguru import logger

from reflectolearn.data_processing.preprocess import preprocess_file
from scripts.config import load_config


def main():
    logger.info("Starting preprocessing")
    config = load_config(Path("config.yml"))
    logger.info(f"Config: {config}")

    raw_file = Path(config["data"]["data_dir"]) / "raw" / config["data"]["file_name"]
    raw_name = Path(config["data"]["file_name"]).stem
    data_version = config["data"]["version"]
    preprocessed_file = (
        Path(config["data"]["data_dir"]) / f"{raw_name}_{data_version}.h5"
    )

    # Preprocess the raw data
    preprocess_file(raw_file, preprocessed_file)


if __name__ == "__main__":
    main()
