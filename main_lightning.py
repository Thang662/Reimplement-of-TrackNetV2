import os
import torch
from dataset_lightning import TennisDataModule
from model_lightning import *
# from train import *
# from utils import *
# from loss import *
from weight_init import weight_init
import argparse
import wandb
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
# import data_setup, train, model_builder, utils

def get_opt():
    parser = argparse.ArgumentParser(description='Train a TrackNetV2 model')
    parser.add_argument('--root', type=str, default='D:/thang/20232/thesis/Dataset/Dataset', help='Path to the root directory of the dataset')
    parser.add_argument('--frame_in', type=int, default=3, help='Number of input frames')
    parser.add_argument('--is_sequential', type=bool, default=True, help='Whether the input frames are sequential')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.04365158322401657, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='TrackNetV2', help='Name of the model')
    parser.add_argument('--experiment_name', type=str, default='tennis', help='Name of the experiment')
    parser.add_argument('--model_save_dir', type=str, default='models', help='Directory to save the model')
    parser.add_argument('--weight_init', action='store_true', help='Whether to use weight initialization')
    parser.add_argument('--NUM_WORKERS', type=int, default=2, help='Number of workers for the DataLoader')
    parser.add_argument('--optimizer', choices=['adam', 'adadelta', 'adamw'], default='adamw', help='Optimizer to use')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory to save the logs')
    parser.add_argument('--wandb_api', type=str, default='', help='API key for Weights & Biases')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--precision', type=str, default='32', help='Precision for training')
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator for training')
    parser.add_argument('--strategy', type=str, default='auto', help='Strategy for training')
    parser.add_argument('--devices', type=str, default='auto', help='Devices for training')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for training')
    parser.add_argument('--log_image_every_n_steps', type=int, default=10, help='Log image every n steps')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = get_opt()

    if opt.seed:
        L.seed_everything(opt.seed)

    if opt.devices != 'auto':
        opt.devices = eval(opt.devices)

    dm = TennisDataModule(root = opt.root, frame_in = opt.frame_in, is_sequential = opt.is_sequential, batch_size = opt.batch_size, num_workers = opt.NUM_WORKERS)

    if opt.wandb_api:
        wandb.login(key = opt.wandb_api)
        wandb_logger = WandbLogger(name = opt.model_name, project = opt.experiment_name)
    tensorboard_logger = TensorBoardLogger('')

    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam
    elif opt.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta
    else:
        optimizer = torch.optim.AdamW

    if opt.weight_init:
        model = LitTrackNetV2(frame_in = opt.frame_in * 3, frame_out = opt.frame_in, optimizer = optimizer, weight_init = weight_init, log_image_every_n_steps = opt.log_image_every_n_steps, lr = opt.learning_rate)
    else:
        model = LitTrackNetV2(frame_in = opt.frame_in * 3, frame_out = opt.frame_in, optimizer = optimizer, log_image_every_n_steps = opt.log_image_every_n_steps, lr = opt.learning_rate)


    # fast_dev_run = True for testing purposes
    # trainer = L.Trainer(fast_dev_run = True) 

    # train on 10 batches, validate on 5 batches (for testing purposes)
    # trainer = L.Trainer(limit_train_batches = 50, limit_val_batches = 5) 

    # find training loop bottleneck
    # trainer = L.Trainer(profiler = 'simple', limit_train_batches = 10, limit_val_batches = 5, max_epochs = 2)
    
    # train model
    # trainer = L.Trainer(max_epochs = opt.num_epochs)
        
    # tune to find learning rate
    # trainer = L.Trainer()
    # tuner = Tuner(trainer)

    # # Run learning rate finder
    # lr_finder = tuner.lr_find(model, datamodule = dm)

    # # Results can be found in
    # print(lr_finder.results)

    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # a = input("Press Enter to continue...")

    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print(new_lr)
    
    # saves top-K checkpoints based on "val_mIoU" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_mIoU",
        mode="max",
        dirpath="models",
        filename= opt.model_name + "-{epoch:02d}-{val_mIoU:.2f}",
    )
    trainer = L.Trainer(max_epochs = opt.num_epochs, callbacks = [checkpoint_callback], logger = [tensorboard_logger, wandb_logger], precision = opt.precision, accelerator = opt.accelerator, strategy = opt.strategy, devices = opt.devices, num_nodes = opt.num_nodes)
    trainer.fit(model, datamodule = dm)