# ReflectoLearn

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

ReflectoLearn is a cutting-edge research project that leverages the power of **Machine Learning (ML)** to revolutionize the analysis of **X-ray Reflectivity (XRR)** data. By combining advanced physical modeling with state-of-the-art ML techniques, this project aims to significantly improve the efficiency, accuracy, and reliability of determining thin film parameters from XRR measurements.

This project is designed for research and experimentation, with a workflow centered around Jupyter notebooks for interactive analysis and Python scripts for reproducible tasks like training and data processing.

## Key Features

*   **Hybrid Modeling Approach:** Integrates physics-based models with deep learning architectures to achieve superior performance.
*   **High-Fidelity Simulations:** Generates realistic XRR data for training and validation, with a focus on accurately modeling experimental factors like roughness and density.
*   **Efficient Fitting Algorithms:** Employs optimized ML algorithms for rapid and robust fitting of XRR curves.
*   **Modular and Extensible:** The codebase is designed to be modular and extensible, allowing for easy integration of new models and algorithms.
*   **Jupyter Notebook-Based Workflow:** Provides a rich, interactive environment for data exploration, model prototyping, and visualization.

## Project Structure

```
ReflectoLearn/
├── notebooks/         # Jupyter notebooks for analysis and experimentation
├── scripts/           # Python scripts for recurring tasks (training, simulation, etc.)
├── src/reflectolearn/ # Core Python library with all the project's logic
├── results/           # Output directory for models, scalers, and stats
├── config.yaml        # Main configuration file
├── pyproject.toml     # Project metadata and dependencies
└── README.md          # This file
```

## Getting Started

### Prerequisites

*   Python 3.12
*   [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ReflectoLearn.git
    cd ReflectoLearn
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    uv pip install -r requirements.txt
    ```

## Workflow

The primary way to interact with this project is through the Jupyter notebooks in the `notebooks` directory. These notebooks are designed to guide you through the process of data exploration, model training, and analysis. The Python scripts in the `scripts` directory are designed to be called from the notebooks or run manually to perform recurring tasks.

For example, to train a model, you would typically open a notebook like `notebooks/model_prototyping.ipynb`, which would then call the `scripts/run_training.py` script with the appropriate configuration.

## Configuration

The project uses a `config.yaml` file to manage all the important parameters, such as file paths, model hyperparameters, and simulation settings. You can create your own configuration file or modify the existing one to suit your needs.

```yaml
project:
  name: "p100o9"
  type: "q4"
  run_id: "${project.name}_${project.type}_${model.name}"
  output_dir: "results/${project.run_id}"

data:
  data_dir: "D:/XRR_AI/hdf5_XRR/data" # Example directory
  input_file: "${data.data_dir}/p100o9.h5"
  output_root: "results/${project.run_id}"

training:
  batch_size: 256
  epochs: 200
  learning_rate: 0.001
  seed: 42

model:
  name: "hybrid"
```

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   This project was inspired by the need for more efficient and accurate XRR data analysis tools.
*   We would like to thank the developers of the open-source libraries used in this project, such as PyTorch, scikit-learn, and refnx.
