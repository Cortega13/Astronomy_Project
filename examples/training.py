#from data_generation import DataGenerator
from ChemSurrogate import data_processing as dp
import gc
import torch
import numpy as np
from ChemSurrogate.trainer import (
    AutoencoderTrainer, 
    EmulatorTrainer,
    load_autoencoder_objects,
    load_emulator_objects
)
from ChemSurrogate.configs import (
    DatasetConfig,
    PredefinedTensors,
    AEConfig,
    EMConfig
)

if __name__ == "__main__":

    # # Firstly generate the uclchem dataset which contains the fixed physical parameters and evolving chemical species abundances.
    # data_generator = DataGenerator(DATASET)
    # data_generator.process_models()
    
    
    # # UCLChem outputs are in CSV format. Convert them to HDF5 format for faster processing.
    # compressor = CSVtoHDF5Compressor(DATASET)
    # compressor.run()
    
    
    # # Generate our training/validation/testing splits, plus lower clip our dataset to 1e-20.
    # dataset_cleaner = DatasetCleanser(DATASET)
    # dataset_cleaner.run()
    
    # If we multiply a vector containing our species by this matrix, we obtain each of the elemental abundances, which are conserved.
    # conservation_matrix_path = DatasetConfig.working_path + "/utils/conservation_matrix.npy"
    # conservation_matrix = dp.generate_conservation_matrix()
    # np.save(conservation_matrix_path, conservation_matrix)
    # print(conservation_matrix)
    
    
    # Train the autoencoder. It's a simple autoencoder which enforces reconstruction and conservation error.
    training_np, validation_np = dp.load_datasets(AEConfig.columns)

    training_np_scaled = dp.abundances_scaling(training_np)
    validation_np_scaled = dp.abundances_scaling(validation_np)
    
    training_t = torch.from_numpy(training_np_scaled).to(torch.float32)
    validation_t = torch.from_numpy(validation_np_scaled).to(torch.float32)
    
    training_Dataset = dp.AutoencoderDataset(training_t)
    validation_Dataset = dp.AutoencoderDataset(validation_t)
    
    del training_np, validation_np, training_np_scaled, validation_np_scaled, training_t, validation_t
    gc.collect()
    
    training_dataloader = dp.tensor_to_dataloader(AEConfig, training_Dataset, is_emulator=False)
    validation_dataloader = dp.tensor_to_dataloader(AEConfig, validation_Dataset, is_emulator=False)
    
    autoencoder, optimizer, scheduler = load_autoencoder_objects()
    
    autoencoder_trainer = AutoencoderTrainer(
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader
        )
    
    autoencoder_trainer.train()
    

    # Train the emulator. It's a MLP which takes timestep, physical parameters, and encoded abundances as input and predicts the abundances at the next timestep.
    # training_np, validation_np = dp.load_datasets(EMConfig.columns)
    # training_dataset = dp.prepare_emulator_dataset(DatasetConfig, AEConfig, training_np)
    # validation_dataset = dp.prepare_emulator_dataset(DatasetConfig, AEConfig, validation_np)
    
    # dp.save_tensors_to_hdf5(training_dataset, category="training")
    # dp.save_tensors_to_hdf5(validation_dataset, category="validation")

    # training_dataset, training_indices = dp.load_tensors_from_hdf5(category="training")
    # validation_dataset, validation_indices = dp.load_tensors_from_hdf5(category="validation")
    
    # training_Dataset = dp.RowRetrievalDataset(training_dataset, training_indices)
    # validation_Dataset = dp.RowRetrievalDataset(validation_dataset, validation_indices)
    # del training_dataset, validation_dataset, training_indices, validation_indices

    # training_dataloader = dp.tensor_to_dataloader(EMConfig, training_Dataset, is_emulator=True)
    # validation_dataloader = dp.tensor_to_dataloader(EMConfig, validation_Dataset, is_emulator=True)
    
    # emulator, autoencoder, optimizer, scheduler = load_objects()
    # emulator_trainer = EmulatorTrainer(
    #     DatasetConfig,
    #     AEConfig,
    #     EMConfig,
    #     emulator,
    #     autoencoder,
    #     optimizer,
    #     scheduler,
    #     training_dataloader,
    #     validation_dataloader
    #     )
    # emulator_trainer.train()
    
    
    # plot_generator = PlotGenerator(DATASET)
    # # Generate a plot showing the mean relative error and its standard deviation for each species at each timestep.
    # plot_generator.plot_model_errors()
    
    # # Generate plots showing the evolution of key species abundances over time.
    # plot_generator.plot_species_evolution(num_models=5)
    
    # # Generate plots showing abundances at time N for randomly sampled physical parameters.
    # plot_generator.plot_parameter_sampling(num_samples=5)
    
    
    # # Benchmarking the speed of the model.
    # benchmark_speed()