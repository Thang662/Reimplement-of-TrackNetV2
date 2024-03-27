"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
from dataset import get_data_loaders
from model import TrackNetV2
from train import *
from utils import *
from loss import *
from weight_init import weight_init
import argparse
import unet2d
# import data_setup, train, model_builder, utils

def get_opt():
    parser = argparse.ArgumentParser(description='Train a TrackNetV2 model')
    parser.add_argument('--root', type=str, default='D:/thang/20232/thesis/Dataset/Dataset', help='Path to the root directory of the dataset')
    parser.add_argument('--frame_in', type=int, default=3, help='Number of input frames')
    parser.add_argument('--is_sequential', type=bool, default=True, help='Whether the input frames are sequential')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='TrackNetV2', help='Name of the model')
    parser.add_argument('--experiment_name', type=str, default='tennis', help='Name of the experiment')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to train the model on')
    parser.add_argument('--model_save_dir', type=str, default='models', help='Directory to save the model')
    parser.add_argument('--weight_init', action='store_true', help='Whether to use weight initialization')
    parser.add_argument('--NUM_WORKERS', type=int, default=2, help='Number of workers for the DataLoader')
    parser.add_argument('--optimizer', choices=['adam', 'adadelta', 'adamw'], default='adamw', help='Optimizer to use')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory to save the logs')
    parser.add_argument('--wandb_api', type=str, default='', help='API key for Weights & Biases')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = get_opt()
    train_dataloader, test_dataloader = get_data_loaders(
        root = opt.root,
        frame_in = opt.frame_in,
        is_sequential = opt.is_sequential,
        batch_size = opt.batch_size,
        NUM_WORKERS = opt.NUM_WORKERS
    )

    print(opt)

    net = TrackNetV2(opt.frame_in * 3,  opt.frame_in).to(opt.device)
    # net = unet2d.TrackNetV2(n_channels = opt.frame_in * 3, n_classes = opt.frame_in)
    # state_dict = torch.load('tracknetv2_tennis_best.pth.tar')
    # net.load_state_dict(state_dict['model_state_dict'], strict = True)
    # net = net.to(opt.device)

    if opt.wandb_api:
        wandb.login(key = opt.wandb_api)
        run = wandb.init(
            project = opt.experiment_name,
            name = opt.model_name,
            config = opt
        )

    if opt.weight_init:
        net.apply(weight_init)

    loss_fn = FocalLoss(gamma = 2)

    if opt.seed:
        seed_everything(seed = opt.seed)

    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = opt.learning_rate)
    elif opt.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(net.parameters())
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr = opt.learning_rate)
    
    train_with_writer(model = net,
                      train_loader = train_dataloader,
                      test_loader = test_dataloader,
                      optimizer = optimizer,
                      experiment_name = opt.experiment_name,
                      model_name = opt.model_name,
                      criterion = loss_fn,
                      run = run if opt.wandb_api else None,
                      epochs = opt.num_epochs,
                      device = opt.device)
    # evaluator = SegmentationMetric(2)
    # test_lost, test_mIoU = test_step(model = net,
    #           test_loader = test_dataloader,
    #           criterion = loss_fn,
    #           evaluator = evaluator,
    #           device = opt.device)
    # print(f"Test Loss: {test_lost:.4e}, Test mIoU: {test_mIoU:.4f}")