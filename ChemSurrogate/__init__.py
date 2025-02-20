# src/__init__.py

# This makes src a package and allows for easier imports

from . import data_processing
from . import configs
from . import analysis
from . import train
from . import nn

# Optional: If you want to expose specific classes/functions for easy access
from .configs import DatasetConfig, AEConfig, EMConfig
from .train import load_objects
