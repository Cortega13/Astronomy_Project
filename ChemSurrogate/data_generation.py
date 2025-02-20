import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from scipy.stats import qmc
try:
    import uclchem
except ImportError:
    print("UCLChem not installed. Data Generation disabled.")

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.working_path = config.working_path
        self.grid_folder = os.path.join(self.working_path, "grid_folder")
        self.samples = np.array()
        self.df = None


    def generate_physical_parameters(self):
        """Generates physical parameter samples using Latin Hypercube Sampling (LHS) in log space."""
        num_samples = self.config.num_models
        param_ranges = self.config.physical_parameter_ranges

        log_min = np.log10([param_ranges[key][0] for key in param_ranges])
        log_max = np.log10([param_ranges[key][1] for key in param_ranges])

        lhs = qmc.LatinHypercube(d=len(param_ranges))
        samples_unit = lhs.random(num_samples)

        sampled_log = log_min + samples_unit * (log_max - log_min)
        self.samples = 10**sampled_log


    def add_output_file(self):
        """Loads samples and creates a DataFrame with output file paths."""
        samples = self.samples
        model_table = pd.DataFrame(samples, columns=["temperature", "density", "radfield", "av"], dtype=np.float32)

        model_table["outputFile"] = model_table.apply(
            lambda row: os.path.join(self.grid_folder, f"{row.temperature}_{row.density}_{row.radfield}_{row.av}.csv"),
            axis=1
        )
        
        os.makedirs(self.grid_folder, exist_ok=True)
        
        self.df = model_table

    def run_model(self, row):
        """Runs the UCLCHEM model for a given parameter set."""
        param_dict = {
            "currentTime": 0,
            "endatfinaldensity": False,
            "freefall": False,
            "initialTemp": row.temperature,
            "initialDens": row.density,
            "radfield": row.radfield,
            "baseAv": row.av,
            "zeta": 2.31,
            "outputFile": row.outputFile,
            "finalTime": 1.0e5,
            "ion": 0,
            "fh": 0.5,
        }
        return uclchem.model.cloud(param_dict=param_dict)[0]

    def process_models(self):
        """Runs the models in parallel using joblib."""        
        print(f"{self.config.num_models} models to run.")
        
        Parallel(n_jobs=10, verbose=100)(
            delayed(self.run_model)(row) for _, row in self.df.iterrows()
        )
