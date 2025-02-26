### Gas-Grain Chemical Network Surrogate

In order to use this surrogate model, refer to the Examples folder which contains notebooks showcasing how to use the models and generate plots.



The dataset for this surrogate model was generated using UCLChem. A dataset generation script is included in this repository to replicate results.

60k models were created for training and 20k for validation. The initial abundances are constant for all models, and only the physical parameters are varied. The physical parameters were sampled using latin hypercube sampling in log-space. The only physical parameter kept constant is the Cosmic Ray Ionization Rate, which is kept at 3.0261e-17 ionizations per second. 

With 100 timesteps per model, the total dataset size is 8,000,000 rows of data. Since the model is trained to predict 1-100 timesteps, we can train using all combinations of pairs of abundances. For example (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), ... etc



The validation loss for the results are defined as

loss = abs(actual - predicted) / actual

This loss is calculated for each species and then the mean is calculated. This is calculated for the entire validation set. The species abundances range from 1e-20 to 1.