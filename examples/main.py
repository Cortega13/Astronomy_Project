#from data_generation import DataGenerator
from ChemSurrogate import data_processing as dp
import gc
import torch
from ChemSurrogate.train import (
    Trainer,
    load_objects,
)
from ChemSurrogate.configs import (
    DatasetConfig,
    ModelConfig,
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
    
    
    # # We minmax scale all the features to (0, 1) for better training for the neural networks.
    # scalars_generator = ScalarsGenerator(DATASET, AE_CONFIG)
    # scalars_generator.autoencoder()
    
    
    # # If we multiply a vector containing our species by this matrix, we obtain each of the elemental abundances, which are conserved.
    # DATASET.conservation_matrix = generate_conservation_matrix(DATASET)
    
    
    # # Train the autoencoder. It's a simple autoencoder which enforces reconstruction and conservation error.
    # autoencoder_trainer = Trainer(DATASET, AE_CONFIG, autoencoder=True)
    
    
    # # We minmax scale the encoded abundances to (0, 1).
    # scalars_generator.emulator()    
    # Train the emulator. It's a MLP which takes timestep, physical parameters, and encoded abundances as input and predicts the abundances at the next timestep.
    training_np, validation_np = dp.load_datasets()
    training_dataset, training_indices = dp.prepare_emulator_dataset(training_np)
    validation_dataset, validation_indices = dp.prepare_emulator_dataset(validation_np)
    
    # dp.save_tensors_to_hdf5(training_dataset, category="training")
    # dp.save_tensors_to_hdf5(validation_dataset, category="validation")
    # training_dataset, training_indices = dp.load_tensors_from_hdf5(category="training")
    # validation_dataset, validation_indices = dp.load_tensors_from_hdf5(category="validation")
    
    training_Dataset = dp.RowRetrievalDataset(training_dataset, training_indices)
    validation_Dataset = dp.RowRetrievalDataset(validation_dataset, validation_indices)
    del training_dataset, validation_dataset, training_indices, validation_indices

    training_dataloader = dp.tensor_to_dataloader(training_Dataset)
    validation_dataloader = dp.tensor_to_dataloader(validation_Dataset)
    
    model, optimizer, scheduler = load_objects()
    emulator_trainer = Trainer(
        model,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader
        )
    emulator_trainer.train()
    
    
    # plot_generator = PlotGenerator(DATASET)
    # # Generate a plot showing the mean relative error and its standard deviation for each species at each timestep.
    # plot_generator.plot_model_errors()
    
    # # Generate plots showing the evolution of key species abundances over time.
    # plot_generator.plot_species_evolution(num_models=5)
    
    # # Generate plots showing abundances at time N for randomly sampled physical parameters.
    # plot_generator.plot_parameter_sampling(num_samples=5)
    
    
    # # Benchmarking the speed of the model.
    # benchmark_speed()