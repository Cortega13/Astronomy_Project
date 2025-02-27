import numpy as np
import os
from joblib import load
import torch
import importlib.resources as pkg_resources
from ChemSurrogate import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DatasetConfig:
    working_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # The path to the root folder of the project.
    num_training_models =   60_000    # Each model has a different set of initial physical parameters. They all begin with identical initial abundances.
    num_validation_models = 20_000  
    num_timesteps_per_model = 100   # Duration that each model runs for. Multiply by timestep_duration to get total evolution time.
    timestep_duration = 1_000       # In years
    num_metadata = 3
    num_physical_parameters = 4
    num_species = 335
    physical_parameter_ranges = {
        "Density":  (1e1, 1e6),       # H nuclei per cm^3. Limits arbitrarily chosen.
        "Radfield": (1e-3, 1e3),     # Habing field. Limits arbitrarily chosen.
        "av":       (1e-2, 1e4),    # Magnitudes. Limits arbitrarily choisen.
        "gasTemp":  (10, 150),      # Kelvin. Grain reactions are too complex under 10 K. Ice mostly sublimates at 150 K and UCLChem sets it as a strict constraint.
    }
    abundances_lower_clipping = np.log10(np.float32(1e-20))   # Abundances are arbitrarily clipped to 1e-20 since anything lower is insignificant.
    abundances_upper_clipping = np.log10(np.float32(1))       # All abundances are relative to number of Hydrogen nuclei. Maximum abundance is all hydrogen in elemental form.

    initial_abundances_path = os.path.join(working_path, "utils/initial_abundances.npy")
    initial_abundances = np.load(initial_abundances_path)
    
    conservation_matrix_path = os.path.join(working_path, "utils/conservation_matrix.npy")
    conservation_matrix = np.load(conservation_matrix_path)
    
    metadata = ["Index", "Model", "Time"]
    physical_parameters = list(physical_parameter_ranges.keys())
    species_path = os.path.join(working_path, "utils/species.txt")
    species = np.loadtxt(species_path, dtype=str, delimiter=" ").tolist()
    
    training_rawdataset_path = os.path.join(working_path, "data/uclchem_rawdata_training.h5")
    validation_rawdataset_path = os.path.join(working_path, "data/uclchem_rawdata_validation.h5")
    
    training_dataset_path = os.path.join(working_path, "data/training.h5")
    validation_dataset_path = os.path.join(working_path, "data/validation.h5")


class ModelConfig:
    window_size = 11
    # Model Config
    input_dim = DatasetConfig.num_species + DatasetConfig.num_physical_parameters
    output_dim = DatasetConfig.num_species
    hidden_dim = 300
    latent_dim = 16
    
    
    # Hyperparameters Config
    atol = 1e-5
    rtol = 1e-2
    lr = 1e-4
    lr_decay = 0.5
    lr_decay_patience = 5
    betas = (0.8, 0.9)
    weight_decay = 0
    loss_scaling_factor = 1e-3
    exponential_coefficient = 22
    alpha = 3e3
    batch_size = 8192
    stagnant_epoch_patience = 20
    max_epochs = 999999
    gradient_clipping = 6
    pretrained_model_path = os.path.join(DatasetConfig.working_path, "models/latentODE.pth")
    save_model_path = os.path.join(DatasetConfig.working_path, "models/latentODE.pth")
    dropout = 0.1
    noise = 0
    save_model = True


class PredefinedTensors:
    ab_min = torch.tensor(DatasetConfig.abundances_lower_clipping, dtype=torch.float32).to(device)
    ab_max = torch.tensor(DatasetConfig.abundances_upper_clipping, dtype=torch.float32).to(device)
    
    conservation_matrix = torch.tensor(DatasetConfig.conservation_matrix, dtype=torch.float32).to(device).contiguous()
    
    exponential = torch.log(torch.tensor(10, dtype=torch.float32)).to(device)

    loss_scaling_factor = torch.tensor(ModelConfig.loss_scaling_factor, dtype=torch.float32).to(device)
    
    exponential_coefficient = torch.tensor(ModelConfig.exponential_coefficient, dtype=torch.float32).to(device)
    
    alpha = torch.tensor(ModelConfig.alpha, dtype=torch.float32).to(device)
    
    mace_max_abundance = torch.tensor(0.85, dtype=torch.float32).to(device)
    mace_factor = torch.tensor(468/335, dtype=torch.float32).to(device)