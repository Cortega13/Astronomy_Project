import numpy as np
import os
from joblib import load
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DatasetConfig:
    """
    THis is general information about the dataset.
    """
    working_path = "C:/Users/carlo/Projects/Astronomy Project/main"
    num_training_models =   60_000      # Each model has a different set of initial physical parameters. They all begin with identical initial abundances.
    num_validation_models = 20_000  
    num_timesteps_per_model = 100       # Duration that each model runs for. Multiply by timestep_duration to get total evolution time.
    timestep_duration = 1_000           # In years
    
    num_metadata = 3
    num_physical_parameters = 4
    num_species = 335
    
    physical_parameter_ranges = {
        "Density":  (1e1, 1e6),         # H nuclei per cm^3. Limits arbitrarily chosen.
        "Radfield": (1e-3, 1e3),        # Habing field. Limits arbitrarily chosen.
        "av":       (1e-2, 1e4),        # Magnitudes. Limits arbitrarily choisen.
        "gasTemp":  (10, 150),          # Kelvin. Grain reactions are too complex under 10 K. Ice mostly sublimates at 150 K and UCLChem sets it as a strict constraint.
    }
    abundances_lower_clipping = np.log10(np.float32(1e-20))   # Abundances are arbitrarily clipped to 1e-20 since anything lower is insignificant.
    abundances_upper_clipping = np.log10(np.float32(1))       # All abundances are relative to number of Hydrogen nuclei. Maximum abundance is all hydrogen in elemental form.

    initial_abundances_path = os.path.join(working_path, "utils/initial_abundances.npy")
    initial_abundances = np.load(initial_abundances_path)
    
    conservation_matrix_path = os.path.join(working_path, "utils/conservation_matrix.npy")
    try:
        conservation_matrix = np.load(conservation_matrix_path)
    except FileNotFoundError:
        print("Conservation Matrix not found. Please generate conservation matrix.")        
    
    metadata = ["Index", "Model", "Time"]
    physical_parameters = list(physical_parameter_ranges.keys())
    species_path = os.path.join(working_path, "utils/species.txt")
    species = np.loadtxt(species_path, dtype=str, delimiter=" ").tolist()
    
    training_rawdataset_path = os.path.join(working_path, "data/uclchem_rawdata_training.h5")
    validation_rawdataset_path = os.path.join(working_path, "data/uclchem_rawdata_validation.h5")
    
    training_dataset_path = os.path.join(working_path, "data/training.h5")
    validation_dataset_path = os.path.join(working_path, "data/validation.h5")


class AEConfig:
    """
    This is the configuration for the Autoencoder Model.
    """
    # Model Config
    input_dim = DatasetConfig.num_species # input_dim = output_dim
    hidden_dim = 600
    latent_dim = 12
    
    # Hyperparameters Config
    lr = 1e-4
    lr_decay = 0.8
    lr_decay_patience = 6
    betas = (0.7, 0.8)
    weight_decay = 5e-5
    loss_scaling_factor = 1e-3
    exponential_coefficient = 26
    alpha = 3e3
    batch_size = 4*8192
    stagnant_epoch_patience = 16
    max_epochs = 999999
    gradient_clipping = 2
    pretrained_model_path = os.path.join(DatasetConfig.working_path, "models/autoencoder.pth")
    save_model_path = os.path.join(DatasetConfig.working_path, "models/auteoncoder.pth")
    dropout = 0.1
    noise = 0.1
    save_model = False
    shuffle = True

    
class EMConfig:
    """
    This is the configuration for the Emulator Model.
    """
    # Model Config
    input_dim = 1 + DatasetConfig.num_physical_parameters + AEConfig.latent_dim # The 1 is for the time input.
    hidden_dim = 600
    output_dim = AEConfig.latent_dim
    
    # Hyperparameters Config
    lr = 1e-4
    lr_decay = 0.8
    lr_decay_patience = 6
    betas = (0.7, 0.8)
    weight_decay = 5e-5
    loss_scaling_factor = 1e-3
    exponential_coefficient = 26
    alpha = 3e3
    batch_size = 4*8192
    stagnant_epoch_patience = 16
    max_epochs = 999999
    gradient_clipping = 2
    pretrained_model_path = os.path.join(DatasetConfig.working_path, "models/emulator.pth")
    save_model_path = os.path.join(DatasetConfig.working_path, "models/emulator.pth")
    dropout = 0.1
    save_model = False
    shuffle = True


class PredefinedTensors:
    """
    These are predefined tensors which are kept on device for speed performance.
    They prevent type casting and moving tensors to and from devices.
    """
    ab_min = torch.tensor(DatasetConfig.abundances_lower_clipping, dtype=torch.float32).to(device)
    ab_max = torch.tensor(DatasetConfig.abundances_upper_clipping, dtype=torch.float32).to(device)
    
    ae_min, ae_max = DatasetConfig.scalers["encoded_components"]
    ae_min = torch.tensor(ae_min, dtype=torch.float32).to(device)
    ae_max = torch.tensor(ae_max, dtype=torch.float32).to(device)
    
    conservation_matrix = torch.tensor(DatasetConfig.conservation_matrix, dtype=torch.float32).to(device).contiguous()
    
    exponential = torch.log(torch.tensor(10, dtype=torch.float32)).to(device)

    loss_scaling_factor = torch.tensor(EMConfig.loss_scaling_factor, dtype=torch.float32).to(device)
    
    exponential_coefficient = torch.tensor(EMConfig.exponential_coefficient, dtype=torch.float32).to(device)
    
    alpha = torch.tensor(EMConfig.alpha, dtype=torch.float32).to(device)
    
    mace_max_abundance = torch.tensor(0.85, dtype=torch.float32).to(device)
    mace_factor = torch.tensor(468/335, dtype=torch.float32).to(device)