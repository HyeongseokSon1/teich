"""Agentic Datagen v2 - Generate training data from Codex and Pi traces."""

__version__ = "2.0.0"

from .config import Config, load_config
from .converter import TrainingExample, convert_trace_to_training_example, convert_traces_to_training_data
from .loader import load_traces

__all__ = [
    "Config",
    "TrainingExample",
    "convert_trace_to_training_example",
    "convert_traces_to_training_data",
    "load_traces",
    "load_config",
]
