import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
from .configs import DatasetConfig
from . import data_processing as dp


def sample_initial_conditions(
    n_samples: int = 100000
    ):
    """
    Generate initial conditions for the parameter sampling plots where physical conditions are varied.
    """
        
    data_rows = []
    for _ in range(n_samples):
        sampled_values = []
        for param in DatasetConfig.physical_parameter_ranges:
            low, high = np.log10(DatasetConfig.physical_parameter_ranges[param])
            val_log10 = np.random.uniform(low, high)
            sampled_values.append(val_log10)

        sampled_values = np.array(sampled_values, dtype=np.float32)
        row = np.hstack((sampled_values, DatasetConfig.initial_abundances.squeeze(0)))
        data_rows.append(row)
    
    data_rows = np.array(data_rows, dtype=np.float32)
    
    physical_parameter_ranges = DatasetConfig.physical_parameter_ranges
    for i, parameter in enumerate(physical_parameter_ranges):
        param_min, param_max = physical_parameter_ranges[parameter]
        log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)
        data_rows[:, i] = (data_rows[:, i] - log_param_min) / (log_param_max - log_param_min)

    inputs = torch.tensor(np.vstack(data_rows), dtype=torch.float32)
    
    return inputs


def add_timesteps_to_conditions(initial_conditions, num_timesteps=95):
    """
    Adds a time column to the initial conditions tensor.
    """
    time_as_fraction = (num_timesteps / DatasetConfig.num_timesteps_per_model)
    batch_size = initial_conditions.shape[0]
    time_column = torch.full((batch_size, 1), time_as_fraction, dtype=initial_conditions.dtype, device=initial_conditions.device)
    
    initial_conditions_with_time = torch.cat((time_column, initial_conditions), dim=1)
    return initial_conditions_with_time


def reconstruct_results(initial_conditions_with_time, decoded_features):
    scaled_physical_parameters = initial_conditions_with_time[:, 1:1+DatasetConfig.num_physical_parameters]
    
    physical_parameters = dp.inverse_physical_parameter_scaling(scaled_physical_parameters)
    
    initial_conditions_with_time
    columns = DatasetConfig.physical_parameters + DatasetConfig.species
    results = torch.cat((physical_parameters, decoded_features.cpu()), dim=1)
    results_df = pd.DataFrame(results.numpy(), columns=columns)
    
    return results_df


def benchmark_speed(DATASET, AE_CONFIG, EMULATOR_CONFIG):
    pass


### Plot Functions
def histogram_physical_parameters(
    config,
    sampled_physical_parameters: np.array
    ):
    """
    Generate histograms for the sampled physical conditions to visualize distribution.
    """
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, param in enumerate(config.physical_parameters):
        row = i // 2
        col = i % 2
        axs[row, col].hist(np.log10(sampled_physical_parameters[:, i]), bins=200, color='steelblue', edgecolor='black')
        axs[row, col].set_title(f"{param} Frequency")
        axs[row, col].set_xlabel("Value")
        axs[row, col].set_ylabel("Count")
    
    plt.tight_layout()
    savefig_path = os.path.join(config.working_path, "plots/histogram_physical_parameters.png")
    plt.savefig(savefig_path, dpi=300, bbox_inches="tight")    


def plot_abundances_vs_time_comparison(
    config,
    original_abundances: pd.DataFrame, 
    reconstructed_abundances: pd.DataFrame, 
    species_of_interest: list, 
    model_num_index: int
    ):
    """
    Plotting the reconstructed and original abundances on a chemical evolution plot.
    This shows how accurate the reconstructed evolution is.
    """
    
    plt.figure(figsize=(10, 6))
    colors = plt.colormaps.get_cmap('tab10')
    for idx, species in enumerate(species_of_interest):
        timesteps = np.arange(0, len(reconstructed_abundances[species]))
        
        plt.plot(
            timesteps, 
            np.log10(original_abundances), 
            label=f"{species} Actual", 
            color=colors(idx), 
            linestyle="-"
        )
        plt.plot(
            timesteps, 
            np.log10(reconstructed_abundances[species]), 
            label=f"{species} Predicted", 
            color=colors(idx), 
            linestyle="--"
        )
    plt.xlabel("Time (x1000 year)")
    plt.ylabel("Log Abundances (Relative to H nuclei)")
    plt.title("Log Abundances vs. Time")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(config.working_path, f"plots/abundances_vs_time_comparison{model_num_index+1}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    

def scatter_abundances_vs_physical_parameters(
    df: pd.DataFrame,
    species_of_interest: list,
    output_folder:str = "plots/scatter_abundances_vs_physical_parameters",
    ):
    """
    Generates a scatter plot for each species and each physical parameter of abundance vs. physical parameter.
    """
    
    global_mins = (np.log10(df[DatasetConfig.physical_parameters].min()))
    global_maxs = (np.log10(df[DatasetConfig.physical_parameters].max()))
    
    for species in species_of_interest:
        _, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, varying_param in enumerate(DatasetConfig.physical_parameters):
            df_subset_sorted = df.sort_values(by=varying_param, ascending=True)
            other_params = [p for p in DatasetConfig.physical_parameters if p != varying_param]
            df_color = np.log10(df_subset_sorted[other_params].astype(float))
            colors = (df_color - global_mins[other_params]) / (
                global_maxs[other_params] - global_mins[other_params]
            )
            colors = (colors - colors.min()) / (colors.max() - colors.min())
            colors = 1 / (1 + np.exp(-10 * (colors - 0.5)))
            colors *= 0.8
            
            colors = colors.to_numpy()

            ax = axes[i]
            ax.scatter(
                np.log10(df_subset_sorted[varying_param]),
                np.log10(df_subset_sorted[species]),
                c=colors,
                marker='.',
                linewidth=0.1,
                label=species
            )

            ax.set_xlabel(f"Log {varying_param}")
            ax.set_ylabel(f"Log {species} Abundance")
            ax.set_title(f"Log {varying_param} vs. Log {species}")
            ax.grid(True)

            channel_info = (
                f"R = {other_params[0]} "
                f"[{global_mins[other_params[0]]:.2e}, {global_maxs[other_params[0]]:.2e}]\n"
                f"G = {other_params[1]} "
                f"[{global_mins[other_params[1]]:.2e}, {global_maxs[other_params[1]]:.2e}]\n"
                f"B = {other_params[2]} "
                f"[{global_mins[other_params[2]]:.2e}, {global_maxs[other_params[2]]:.2e}]"
            )
            ax.text(
                0., 1,
                channel_info,
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
            )

        plt.tight_layout()
        folder_path = os.path.join(DatasetConfig.working_path, output_folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        savefig_path = os.path.join(folder_path, f"{species}.png")
        plt.savefig(savefig_path, dpi=300, bbox_inches="tight")
        plt.show()
### Plot Functions




### Statistics Functions
def calculate_conservation_error(
    tensor1: torch.Tensor, 
    tensor2: torch.Tensor
    ):
    """
    returns
    mean_original_elemental_abundances, mean_reconstruction_elemental_abundances, conservation_error
    
    """
    # Calculates the conservation error.
    
    unscaled_tensor1 = check_conservation(tensor1).mean(dim=0, keepdim=True)
    unscaled_tensor2 = check_conservation(tensor2).mean(dim=0, keepdim=True)
    conservation_error = abs(unscaled_tensor1 - unscaled_tensor2) / unscaled_tensor1
    return unscaled_tensor1, unscaled_tensor2, conservation_error.mean().item()


def calculate_mace_error(
    original_abundances: torch.Tensor, 
    reconstructed_abundances: torch.Tensor
    ):
    """
    This is the error function defined in the MACE github repository.
    """
    
    original_abundances_np = original_abundances.cpu().numpy()
    reconstructed_abundances_np = reconstructed_abundances.cpu().numpy()
    
    ### Optional Clipping.
    # In the MACE repo, their maximum abundance is defined as 0.85.
    mace_maximum_abundance = 0.85
    
    original_abundances_np = np.clip(original_abundances_np, 0, mace_maximum_abundance, dtype=np.float32)
    reconstructed_abundances_np = np.clip(reconstructed_abundances_np, 0, mace_maximum_abundance, dtype=np.float32)
    
    mace_species_error = (np.log10(original_abundances_np[:])-np.log10(reconstructed_abundances_np))[:]/np.log10(reconstructed_abundances_np[:][:])
    num_samples = len(original_abundances_np[:,0])

    # Since MACE has 468 species, and we have 335, we multiply our error by 468/335 for a fair comparison.
    mace_error = np.abs(mace_species_error).sum()/num_samples * (468 / 334)
    return mace_species_error, mace_error


def calculate_relative_error(
    original_abundances: torch.Tensor, 
    reconstructed_abundances: torch.Tensor
    ):
    """
    Returns
    mean_relative_error, std_relative_error, highest_20_species_mean_relative_error
    """
    
    relative_error = (abs((original_abundances - reconstructed_abundances)) / original_abundances).mean(dim=0)
    sorted_errors, _ = torch.sort(relative_error, descending=True)
    
    return relative_error.mean(), relative_error.std(), sorted_errors[:20]
### Statistics Functions


def results_analysis(outputs, decoded_outputs):
    mace_species_error, mace_error = calculate_mace_error(outputs, decoded_outputs)
    mean_relative_error, std_relative_error, highest_species_error = calculate_relative_error(outputs, decoded_outputs)
    mean_original_elemental_abundances, mean_reconstruction_elemental_abundances, conservation_error = calculate_conservation_error(decoded_outputs, outputs)

    print(f"Conservation Error: {conservation_error:.3e}")
    print(f"MACE Error: {mace_error:.4e}")
    print(f"Mean Relative Error: {mean_relative_error:.3e} | Std Relative Error: {std_relative_error:.3e}")
    print()