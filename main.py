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
# import data_setup, train, model_builder, utils

def get_opt():
    parser = argparse.ArgumentParser(description='Train a TrackNetV2 model')
    parser.add_argument('--root', type=str, default='D:/thang/20232/Dataset/Dataset', help='Path to the root directory of the dataset')
    parser.add_argument('--frame_in', type=int, default=3, help='Number of input frames')
    parser.add_argument('--is_sequential', type=bool, default=True, help='Whether the input frames are sequential')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='TrackNetV2', help='Name of the model')
    parser.add_argument('--experiment_name', type=str, default='tennis', help='Name of the experiment')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to train the model on')
    parser.add_argument('--model_save_dir', type=str, default='models', help='Directory to save the model')
    parser.add_argument('--weight_intit', action='store_true', help='Whether to use weight initialization')
    parser.add_argument('--NUM_WORKERS', type=int, default=2, help='Number of workers for the DataLoader')
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

    net = TrackNetV2(in_channels = opt.frame_in * 3, out_channels = opt.frame_in).to(opt.device)
    if opt.weight_intit:
        net.apply(weight_init)
    
    loss_fn = FocalLoss(gamma = 2)
    optimizer = torch.optim.Adam(net.parameters(),
                                lr = opt.learning_rate)
    
    train_with_writer(model = net,
                      train_loader = train_dataloader,
                      test_loader = test_dataloader,
                      optimizer = optimizer,
                      experiment_name = opt.experiment_name,
                      model_name = opt.model_name,
                      criterion = loss_fn,
                      epochs = opt.num_epochs,
                      device = opt.device)
    
    save_model(model=net,
               target_dir=opt.model_save_dir,
               model_name=f"{opt.model_name}.pth")
    # # Setup hyperparameters
    # NUM_EPOCHS = 30
    # BATCH_SIZE = 2
    # LEARNING_RATE = 0.001
    # frame_in = 3
    # is_sequential = True


    # # Setup directories
    # root = 'Dataset/Dataset'

    # # Setup target device
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # # # Create transforms
    # # data_transform = transforms.Compose([
    # #   transforms.Resize((64, 64)),
    # #   transforms.ToTensor()
    # # ])

    # # Create DataLoaders with help from data_setup.py
    # train_dataloader, test_dataloader = get_data_loaders(
    #     root = root,
    #     frame_in = frame_in,
    #     is_sequential = is_sequential,
    #     batch_size = BATCH_SIZE,
    #     NUM_WORKERS = 2
    # )

    # # Get class names
    # # class_names = train_dataloader.dataset.classes

    # # Create model with help from model_builder.py
    # net = TrackNetV2(in_channels = frame_in * 3, out_channels = frame_in).to(device)
    # net.apply(weight_init)

    # # Set loss and optimizer
    # loss_fn = FocalLoss(gamma = 2)
    # optimizer = torch.optim.Adam(net.parameters(),
    #                             lr = LEARNING_RATE)

    # # Start training with help from engine.py
    # train_with_writer(model = net,
    #             train_loader = train_dataloader,
    #             test_loader = test_dataloader,
    #             criterion = loss_fn,
    #             optimizer = optimizer,
    #             epochs = NUM_EPOCHS,
    #             device = device,
    #             experiment_name = 'tennis',
    #             model_name = 'TrackNetV2')

    # # Save the model with help from utils.py
    # save_model(model=net,
    #             target_dir="models",
    #             model_name="TrackNetV2.pth")