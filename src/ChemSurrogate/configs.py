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
    num_species = 333
    physical_parameter_ranges = {
        "Density":  (1e1, 1e6),       # H nuclei per cm^3. Limits arbitrarily chosen.
        "Radfield": (1e-3, 1e3),     # Habing field. Limits arbitrarily chosen.
        "av":       (1e-2, 1e4),    # Magnitudes. Limits arbitrarily choisen.
        "gasTemp":  (10, 150),      # Kelvin. Grain reactions are too complex under 10 K. Ice mostly sublimates at 150 K and UCLChem sets it as a strict constraint.
    }
    abundances_lower_clipping = np.float32(1e-20)   # Abundances are arbitrarily clipped to 1e-20 since anything lower is insignificant.
    abundances_upper_clipping = np.float32(1)       # All abundances are relative to number of Hydrogen nuclei. Maximum abundance is all hydrogen in elemental form.

    initial_abundances_path = os.path.join(working_path, "utils/initial_abundances.npy")
    initial_abundances = np.load(initial_abundances_path)
    
    stoichiometric_matrix_path = os.path.join(working_path, "utils/stoichiometric_matrix.npy")
    stoichiometric_matrix = np.load(stoichiometric_matrix_path)
        
    metadata = ["Index", "Model", "Time"]
    physical_parameters = list(physical_parameter_ranges.keys())
    species_path = os.path.join(working_path, "utils/species.txt")
    species = np.loadtxt(species_path, dtype=str, delimiter=" ").tolist()
    
    training_dataset_path = os.path.join(working_path, "data/uclchem_training.h5")
    validation_dataset_path = os.path.join(working_path, "data/uclchem_validation.h5")

class AEConfig:
    columns = DatasetConfig.species
    num_columns = len(columns)
    component_scalers_path = os.path.join(DatasetConfig.working_path, "utils/component_scalers.npy")
    # Model Config
    input_dim = DatasetConfig.num_species # input_dim = output_dim
    hidden_dim = 600
    latent_dim = 12
    
    # Hyperparameters Config
    lr = 6e-5
    lr_decay = 0.5
    lr_decay_patience = 20
    betas = (0.6, 0.7)
    weight_decay = 0
    loss_scaling_factor = 1e-3
    exponential_coefficient = 36
    alpha = 1e2
    batch_size = 8192
    stagnant_epoch_patience = 20
    gradient_clipping = 4
    pretrained_model_path = os.path.join(DatasetConfig.working_path, "models/autoencoder.pth")
    save_model_path = os.path.join(DatasetConfig.working_path, "models/autoencoder.pth")
    noise = 0.1
    save_model = False

    
class EMConfig:
    columns = DatasetConfig.metadata + DatasetConfig.physical_parameters + DatasetConfig.species
    num_columns = len(columns)
    # Model Config
    input_dim = DatasetConfig.num_physical_parameters + AEConfig.latent_dim + 1 # The 1 is for the time input.
    hidden_dim = 300
    output_dim = AEConfig.latent_dim
    
    # Hyperparameters Config
    lr = 1e-3
    lr_decay = 0.6
    lr_decay_patience = 3
    betas = (0.4, 0.5)
    weight_decay = 2e-5
    loss_scaling_factor = 1e-3
    exponential_coefficient = 18
    alpha = 1e3
    batch_size = 4*8192
    stagnant_epoch_patience = 20
    gradient_clipping = 1
    pretrained_model_path = os.path.join(DatasetConfig.working_path, "models/emulator.pth")
    save_model_path = os.path.join(DatasetConfig.working_path, "models/emulator.pth")
    dropout = 0.14
    save_model = True
    shuffle = True


class PredefinedTensors:
    ab_min = torch.tensor(np.log10(DatasetConfig.abundances_lower_clipping), dtype=torch.float32).to(device)
    ab_max = torch.tensor(np.log10(DatasetConfig.abundances_upper_clipping), dtype=torch.float32).to(device)

    component_scalers_path = os.path.join(DatasetConfig.working_path, "utils/component_scalers.npy")
    ae_min, ae_max = np.load(component_scalers_path)
    ae_min = torch.tensor(ae_min, dtype=torch.float32).to(device)
    ae_max = torch.tensor(ae_max, dtype=torch.float32).to(device)
    
    stoichiometric_matrix = torch.tensor(DatasetConfig.stoichiometric_matrix, dtype=torch.float32).to(device).contiguous()
    
    exponential = torch.log(torch.tensor(10, dtype=torch.float32)).to(device)

    AE_loss_scaling_factor = torch.tensor(AEConfig.loss_scaling_factor, dtype=torch.float32).to(device)
    EM_loss_scaling_factor = torch.tensor(EMConfig.loss_scaling_factor, dtype=torch.float32).to(device)
    
    AE_exponential_coefficient = torch.tensor(AEConfig.exponential_coefficient, dtype=torch.float32).to(device)
    EM_exponential_coefficient = torch.tensor(EMConfig.exponential_coefficient, dtype=torch.float32).to(device)
    
    AE_alpha = torch.tensor(AEConfig.alpha, dtype=torch.float32).to(device)
    EM_alpha = torch.tensor(EMConfig.alpha, dtype=torch.float32).to(device)
    
    mace_max_abundance = torch.tensor(0.85, dtype=torch.float32).to(device)
    mace_factor = torch.tensor(468/335, dtype=torch.float32).to(device)