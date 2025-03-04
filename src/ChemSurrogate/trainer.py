import os
import gc
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from .nn import Autoencoder, Emulator
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn

from . import data_processing as dp
from .configs import DatasetConfig, AEConfig, EMConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer:
    def __init__(
        self,
        model_config,
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
        self.model_config = model_config
        
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
        model_path = os.path.join(self.model_config.save_model_path)
        if self.model_config.save_model:
            torch.save(checkpoint, model_path)


    def _check_early_stopping(self):
        """
        Ends training once the number of stagnant epochs exceeds the patience.
        """
        if self.stagnant_epochs >= self.model_config.stagnant_epoch_patience:
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
        max_loss = val_loss.max().item()
        metric = mean_loss + std_loss + 0.5*max_loss
        
        if metric < self.metric_minimum_loss:
            print("**********************")
            print(f"New Minimum \nMean: {mean_loss:.3e} \nStd: {std_loss:.3e} \nMax: {max_loss:.3e} \nMetric: {metric:.3e} \nPercent Improvement: {(100-metric*100/self.metric_minimum_loss):.3f}%")
            self._save_checkpoint()
            
            self.metric_minimum_loss = metric
            self.stagnant_epochs = 0
            self.loss_per_epoch.append(metric)
        else:
            self.stagnant_epochs += 1
            print(f"Stagnant {self.stagnant_epochs} \nMinimum: {self.metric_minimum_loss:.3e} \nMean: {mean_loss:.3e} \nStd: {std_loss:.3e} \nMax: {max_loss:.3e}")
        
        self.epoch_validation_loss.zero_()
        self.scheduler.step(metric)
        print()
        print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.3e}")


class AutoencoderTrainer(Trainer):
    def __init__(
        self,
        autoencoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """
        Initializes the AutoencoderTrainer, a subclass of Trainer, specialized for training the autoencoder.
        """        
        self.num_metadata = DatasetConfig.num_metadata
        self.num_physical_parameters = DatasetConfig.num_physical_parameters
        self.num_species = DatasetConfig.num_species
        self.num_components = AEConfig.latent_dim
        
        self.ae = autoencoder
        
        super().__init__(
            model_config=AEConfig,
            model=autoencoder,
            optimizer=optimizer,
            scheduler=scheduler,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
        )

    def _run_training_batch(self, features):
        """
        Runs a training batch where features = targets since this is an autoencoder.
        """
        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = dp.autoencoder_loss_function(outputs, features)
                
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), AEConfig.gradient_clipping)
        self.optimizer.step()


    def _run_validation_batch(self, features):
        """
        Runs a validation batch where features = targets since this is an autoencoder.
        """
        component_outputs = self.model.encode(features)
        outputs = self.model.decode(component_outputs)

        loss = dp.validation_loss_function(outputs, features)
        self.epoch_validation_loss += loss


    def _run_epoch(self, epoch):
        """
        Since this is an autoencoder, there are no targets and thus the dataloaderss only have features.
        """
        self.training_dataloader.sampler.set_epoch(epoch)
        
        tic1 = datetime.now()
        self.model.train()
        for features in self.training_dataloader:
            features = features[0].to(device, non_blocking=True)
            self._run_training_batch(features)

        tic2 = datetime.now()
        self.model.eval()
        with torch.no_grad():
            for features in self.validation_dataloader:
                features = features[0].to(device, non_blocking=True)
                self._run_validation_batch(features)

        toc = datetime.now()
        print(f"Training Time: {tic2 - tic1} | Validation Time: {toc - tic2}\n")


    def train(self):
        """
        Training loop for the autoencoder. Runs until the minimum loss stagnates for a number of epochs.
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        for epoch in range(999999):
            self._run_epoch(epoch)
            self._check_minimum_loss()
            if self._check_early_stopping():
                break

        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nTraining Complete. Trial Results: {self.metric_minimum_loss}")


class EmulatorTrainer(Trainer):
    def __init__(
        self,
        emulator: torch.nn.Module,
        autoencoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """
        Initializes the EmulatorTrainer, a subclass of Trainer, specialized for train the emulator.
        """
        self.ae = autoencoder
        
        super().__init__(
            model_config=EMConfig,
            model=emulator,
            optimizer=optimizer,
            scheduler=scheduler,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
        )


    def _run_training_batch(self, features, targets):
        """
        Runs a single training batch.
        """
        self.optimizer.zero_grad()
        outputs = self.model(features)
        outputs = dp.inverse_latent_components_scaling(outputs)
        outputs = self.ae.decode(outputs)
                
        loss = dp.emulator_training_loss_function(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), EMConfig.gradient_clipping)
        self.optimizer.step()


    def _run_validation_batch(self, features, targets):
        """
        Runs a single validation batch.
        """
        outputs = self.model(features)
        outputs = dp.inverse_latent_components_scaling(outputs)
        outputs = self.ae.decode(outputs)
        
        loss = dp.validation_loss_function(outputs, targets)
        
        self.epoch_validation_loss += loss


    def _run_epoch(self, epoch):
        """
        Runs a single epoch of training and validation.
        """        
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
        
        for epoch in range(999999):
            self._run_epoch(epoch)
            self._check_minimum_loss()
            if self._check_early_stopping():
                break
        
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nTraining Complete. Trial Results: {self.metric_minimum_loss}")


def load_autoencoder_objects(is_inference=False):
    ae = Autoencoder(
        input_dim=AEConfig.input_dim,
        latent_dim=AEConfig.latent_dim,
        hidden_dim=AEConfig.hidden_dim,
        noise=0.0 if is_inference else AEConfig.noise,
    ).to(device)
    if os.path.exists(AEConfig.pretrained_model_path):
        print("Loading Pretrained Model")
        ae.load_state_dict(torch.load(AEConfig.pretrained_model_path))

    if is_inference:
        ae.eval()
        for param in ae.parameters():
            param.requires_grad = False
    

    optimizer = optim.AdamW(
        ae.parameters(),
        lr=AEConfig.lr,
        betas=AEConfig.betas,
        weight_decay=AEConfig.weight_decay,
        fused=True,
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=AEConfig.lr_decay,
        patience=AEConfig.lr_decay_patience,
    )

    return ae, optimizer, scheduler


def load_emulator_objects(is_inference=False):
    ae = Autoencoder(
        input_dim=AEConfig.input_dim,
        latent_dim=AEConfig.latent_dim,
        hidden_dim=AEConfig.hidden_dim,
    ).to(device)
    if os.path.exists(AEConfig.pretrained_model_path):
        ae.load_state_dict(torch.load(AEConfig.pretrained_model_path))
    
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False

    emulator = Emulator(
        input_dim=EMConfig.input_dim,
        output_dim=EMConfig.output_dim,
        hidden_layer=EMConfig.hidden_dim,
        dropout=0.0 if is_inference else EMConfig.dropout,
    ).to(device)
    if os.path.exists(EMConfig.pretrained_model_path):
        print("Loading Pretrained Model")
        emulator.load_state_dict(torch.load(EMConfig.pretrained_model_path))

    if is_inference:
        emulator.eval()
        for param in emulator.parameters():
            param.requires_grad = False
    
    optimizer = optim.AdamW(
        emulator.parameters(),
        lr=EMConfig.lr,
        betas=EMConfig.betas,
        weight_decay=EMConfig.weight_decay,
        fused=True,
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=EMConfig.lr_decay,
        patience=EMConfig.lr_decay_patience,
    )

    return emulator, ae, optimizer, scheduler