"""
This module provides specific classes for managing configuration parameters of an experiment.
It builds upon the generic configuration management provided by the base_config module.
"""

from pathlib import Path

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """
    Configuration for the project, including run ID and output directory.
    """

    name: str = Field(..., description="Name of the project")
    type: str = Field(..., description="Type of the project")
    run_id: str = Field(..., description="Unique identifier for the experiment run")
    output_dir: Path = Field(..., description="Directory where results will be saved")


class DataConfig(BaseModel):
    """
    Configuration for data paths and related parameters.
    """

    data_dir: Path = Field(..., description="Directory containing the input data files")
    input_file: Path = Field(..., description="Path to the input data file")


class TrainingConfig(BaseModel):
    """
    Configuration for training parameters.
    """

    batch_size: int = Field(256, description="Batch size for training")
    epochs: int = Field(100, description="Number of epochs for training")
    learning_rate: float = Field(0.001, description="Learning rate for the optimizer")
    seed: int = Field(42, description="Random seed for reproducibility")
    patience: int = Field(50, description="Number of epochs to wait for improvement before stopping training (Early Stopping)")


class ModelConfig(BaseModel):
    """
    Configuration for the model architecture and parameters.
    """

    name: str = Field(..., description="Type of the model (e.g., 'hybrid', 'cnn')")


class ExpConfig(BaseModel):
    """
    Main configuration class for the experiment, combining all other configurations.
    """

    project: ProjectConfig = Field(..., description="Project configuration")
    data: DataConfig = Field(..., description="Data configuration")
    training: TrainingConfig = Field(..., description="Training configuration")
    model: ModelConfig = Field(..., description="Model configuration")
