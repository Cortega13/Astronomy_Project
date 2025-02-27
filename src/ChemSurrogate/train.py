import os
import gc
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from .nn import LatentODEFunction, LatentODE
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn
import torchode as tode


from . import data_processing as dp
from .configs import DatasetConfig, ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """
        Initializes the Trainer class. A class which simplifies training by including all necessary components.
        """
        self.start_time = datetime.now()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.num_validation_batches = len(self.validation_dataloader.dataset)
                
        self.metric_minimum_loss = np.inf
        self.epoch_validation_loss = torch.zeros(
            DatasetConfig.num_species
        ).to(device)
        self.stagnant_epochs = 0
        self.loss_per_epoch = []


    def _save_checkpoint(self):
        """
        Saves the model's state dictionary to a file.
        """
        checkpoint = self.model.state_dict()
        model_path = os.path.join(ModelConfig.save_model_path)
        if ModelConfig.save_model:
            torch.save(checkpoint, model_path)


    def _check_early_stopping(self):
        """
        Ends training once the number of stagnant epochs exceeds the patience.
        """
        if self.stagnant_epochs >= ModelConfig.stagnant_epoch_patience:
            print("Ending training early due to stagnant epochs.")
            return True
        return False


    def _check_minimum_loss(self):
        """
        Checks if the current epoch's validation loss is the minimum loss so far.
        Calculates the mean relative error and the std of the species-wise mean relative error.
        Uses a metric for the minimum loss which gives weight to the mean and std relative errors.
        Includes a scheduler to reduce the learning rate once the minimum loss stagnates.
        """
        val_loss = self.epoch_validation_loss / self.num_validation_batches
        mean_loss = val_loss.mean().item()
        std_loss = val_loss.std().item()
        metric = mean_loss# + std_loss
        
        if metric < self.metric_minimum_loss:
            print(f"New Minimum | Mean: {mean_loss:.3e} | Std: {std_loss:.3e} | Max: {val_loss.max():.3e}| Metric: {metric:.3e} | Percent Improvement: {(100-metric*100/self.metric_minimum_loss):.3f}%")
            self._save_checkpoint()
            
            self.metric_minimum_loss = metric
            self.stagnant_epochs = 0
            self.loss_per_epoch.append(metric)
        else:
            self.stagnant_epochs += 1
            print(f"Stagnant: {self.stagnant_epochs} | Minimum: {self.metric_minimum_loss:.3e} | Mean: {mean_loss:.3e} | Std: {std_loss:.3e} | Max: {val_loss.max():.3e}")
        
        self.epoch_validation_loss.zero_()
        self.scheduler.step(metric)
        print()


    def _run_training_batch(self, features, targets):
        """
        Runs a single training batch.
        """
        self.optimizer.zero_grad(set_to_none=True)
        
        
        timesteps = torch.linspace(0, 1, steps=DatasetConfig.num_timesteps_per_model+1)[1:targets.size(1)+1].to(device)
        timesteps = timesteps.unsqueeze(0).repeat(targets.size(0), 1)
        
        physical_parameters = features[:, :DatasetConfig.num_physical_parameters]
        species = features[:, DatasetConfig.num_physical_parameters:]
        
        outputs = self.model(timesteps, physical_parameters, species, is_training=True)
                
        loss = dp.emulator_training_loss_function(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), ModelConfig.gradient_clipping)
        self.optimizer.step()


    def _run_validation_batch(self, features, targets):
        """
        Runs a single validation batch.
        """
        timesteps = torch.linspace(0, 1, steps=DatasetConfig.num_timesteps_per_model+1)[1:targets.size(1)+1].to(device)
        timesteps = timesteps.unsqueeze(0).repeat(targets.size(0), 1)
        
        physical_parameters = features[:, :DatasetConfig.num_physical_parameters]
        species = features[:, DatasetConfig.num_physical_parameters:]
        
        outputs = self.model(timesteps, physical_parameters, species, is_training=True)
                
        loss = dp.emulator_validation_loss_function(outputs, targets).mean(dim=0)
        self.epoch_validation_loss += loss


    def _run_epoch(self, epoch):
        """
        Runs a single epoch of training and validation.
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        self.training_dataloader.sampler.set_epoch(epoch)
        
        tic1 = datetime.now()
        self.model.train()
        for features, targets in self.training_dataloader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            self._run_training_batch(features, targets)

        tic2 = datetime.now()
        self.model.eval()
        with torch.no_grad():
            for features, targets in self.validation_dataloader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                self._run_validation_batch(features, targets)
        
        toc = datetime.now()
        print(f"Training Time: {tic2 - tic1} | Validation Time: {toc - tic2}")


    def train(self):
        """
        Training loop for the model. Runs until the minimum loss stagnates for a number of epochs.
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        for epoch in range(9999999):
            self._run_epoch(epoch)
            self._check_minimum_loss()
            if self._check_early_stopping():
                break
        
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nTraining Complete. Trial Results: {self.metric_minimum_loss}")


def load_objects(is_inference=False):

    ode_func = LatentODEFunction(ModelConfig.latent_dim+4)
    
    odeterm = tode.ODETerm(ode_func, with_args=False)
    step_method = tode.Dopri5(term=odeterm)
    step_size_controller = tode.IntegralController(
        atol = ModelConfig.atol,
        rtol = ModelConfig.rtol,
        term = odeterm
    )
    adjoint = tode.AutoDiffAdjoint(step_method, step_size_controller).to(device)
    jit_solver = torch.compile(adjoint)
    
    model = LatentODE(
        input_dim = ModelConfig.output_dim,
        hidden_dim = ModelConfig.hidden_dim,
        latent_dim = ModelConfig.latent_dim,
        output_dim = ModelConfig.output_dim,
        jit_solver = jit_solver,
        dropout = ModelConfig.dropout,
    ).to(device)
    
    if os.path.exists(ModelConfig.pretrained_model_path):
        print("Loading Pretrained Model")
        model.load_state_dict(torch.load(ModelConfig.pretrained_model_path, weights_only=True))

    
    if is_inference:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    optimizer = optim.AdamW(
        model.parameters(),
        lr=ModelConfig.lr,
        betas=ModelConfig.betas,
        weight_decay=ModelConfig.weight_decay,
        fused=True,
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=ModelConfig.lr_decay,
        patience=ModelConfig.lr_decay_patience,
    )

    return model, optimizer, scheduler