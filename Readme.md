Validation Loss Definition

loss = abs(actual - predicted) / actual

This loss is calculated for each species and then the mean is calculated. This is calculated for the entire validation set. The species abundances range from 1e-20 to 1. 