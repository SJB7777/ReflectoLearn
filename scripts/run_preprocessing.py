from loguru import logger

from reflectolearn.config import ConfigManager
from reflectolearn.processing.preprocess import preprocess_file


def main():
    logger.info("Starting preprocessing")
    ConfigManager.initialize("config.yaml")
    config = ConfigManager.load_config()
    logger.info(f"Config: {config}")

    raw_file = config.path.data_file
    raw_name = raw_file.stem
    project_version = config.project.version
    preprocessed_file = config.path.data_root / f"{raw_name}_{project_version}.h5"

    # Preprocess the raw data
    preprocess_file(raw_file, preprocessed_file)


if __name__ == "__main__":
    main()
