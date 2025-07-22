from pathlib import Path

from reflectolearn.data_processing.simulate import make_n_layer_structure
from reflectolearn.config import load_config


def main():
    config: dict[str, dict] = load_config(Path("config.yml"))
    save_file = Path(config["data"]["data_dir"]) / f"{config["data"]["file_name"]}_{config["data"]["version"]}.h5"
    print(f"Saving to {save_file}")


if __name__ == "__main__":
    main()
