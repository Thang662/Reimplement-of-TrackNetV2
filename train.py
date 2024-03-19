import torch
from torch import nn
from tqdm.auto import tqdm
from utils import create_writer, save_model
from postprocessor import run
import matplotlib.pyplot as plt
import cv2
import os
from score import SegmentationMetric

def train_step(model: nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion, evaluator, epoch = 1, device: str = 'cuda') -> tuple[float, float]:
    """
    Performs a single training step.
    Args:
        model (torch.nn.Module): A PyTorch module.
        X (torch.Tensor): A batch of input data.
        y (torch.Tensor): A batch of target data.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer.
        criterion (torch.nn.Module): A PyTorch loss function.
        device (str): Device to train the model on.
    """
    pbar = tqdm(total = len(train_loader.dataset))
    model.train()
    train_loss = 0
    os.makedirs('hm', exist_ok=True)
    for batch, (imgs, heatmaps, annos, annos_transformed) in enumerate(train_loader):
        # Move tensors to the configured device.
        imgs, heatmaps = imgs.to(device), heatmaps.to(device)

        # Forward pass.
        logits = model(imgs)

        # Compute and accumlate loss.
        loss = criterion(logits, heatmaps)
        train_loss += loss.item()
        # preds = logits.clone().detach()
        # results = run(preds)
        # if loss < 0.16:
        #     print(results[0][0])
        #     print(annos_transformed[0][0])

        # Calculate and accumulate mIoU metric across all batches
        preds = torch.sigmoid(logits)
        evaluator.update(preds, heatmaps)
        mIoU = evaluator.get()

        # Backward pass.
        loss.backward()

        # Update parameters.
        optimizer.step()
        optimizer.zero_grad()
        
        # Calculate and accumulate accuracy metric across all batches
        if batch % 100 == 0:
            preds = logits.detach()
            preds = torch.sigmoid(preds).cpu().numpy()
            plt.subplot(3, 2, 1)
            th, hm_th        = cv2.threshold(preds[0][0], 0.5, 1, cv2.THRESH_BINARY)
            plt.imshow(hm_th)
            plt.subplot(3, 2, 2)
            plt.imshow(heatmaps[0][0].cpu().numpy())
            plt.subplot(3, 2, 3)
            th, hm_th        = cv2.threshold(preds[0][1], 0.5, 1, cv2.THRESH_BINARY)
            plt.imshow(hm_th)
            plt.subplot(3, 2, 4)
            plt.imshow(heatmaps[0][1].cpu().numpy())
            plt.subplot(3, 2, 5)
            th, hm_th        = cv2.threshold(preds[0][2], 0.5, 1, cv2.THRESH_BINARY)
            plt.imshow(hm_th)
            plt.subplot(3, 2, 6)
            plt.imshow(heatmaps[0][2].cpu().numpy())
            plt.savefig(f"hm/hm_{batch}_{epoch}.png")

        # Update progress bar.
        pbar.update(len(imgs))
        pbar.set_postfix({'Train Loss': train_loss / (batch + 1), 'mIoU': mIoU})

    # Compute average loss and accuracy across all batches.
    train_loss, train_mIoU = train_loss / len(train_loader), mIoU
    return train_loss, train_mIoU

def test_step(model: nn.Module, test_loader: torch.utils.data.DataLoader, criterion, evaluator, device: str = 'cuda') -> tuple[float, float]:
    """
    Performs a single testing step.
    Args:
        model (torch.nn.Module): A PyTorch module.
        X (torch.Tensor): A batch of input data.
        y (torch.Tensor): A batch of target data.
        criterion (torch.nn.Module): A PyTorch loss function.
        device (str): Device to train the model on.
    """
    model.eval()
    test_loss, test_mIoU = 0, 0
    with torch.inference_mode():
        for batch, (imgs, heatmaps, annos, annos_transformed) in enumerate(test_loader):
            # Move tensors to the configured device.
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)

            # Forward pass.
            logits = model(imgs)

            # Compute and accumlate loss.
            loss = criterion(logits, heatmaps)
            test_loss += loss.item()

            # Calculate and accumulate mIoU metric across all batches
            preds = torch.sigmoid(logits)
            evaluator.update(preds, heatmaps)

        # Compute average loss and accuracy across all batches.
        test_loss, test_mIoU = test_loss / len(test_loader), evaluator.get()
    return test_loss, test_mIoU
    
def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module = nn.CrossEntropyLoss(), epochs: int = 5, device: str = 'cuda') -> dict[str, list[float]]:
    """
    Trains a PyTorch model.
    Args:
        module (torch.nn.Module): A PyTorch module.
        train_loader (torch.utils.data.DataLoader): A PyTorch DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): A PyTorch DataLoader for the testing dataset.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer.
        criterion (torch.nn.Module): A PyTorch loss function.
        epochs (int): Number of epochs to train the model for.
        device (str): Device to train the model on.

    Returns:
        dict[str, list[float]]: A dictionary containing the training and testing loss and accuracy for each epoch.
    """
    results = {'train_loss': [], 'train_mIoU': [], 'test_loss': [], 'test_mIoU': []}
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        train_loss, train_mIoU = train_step(model, train_loader, optimizer, criterion, epoch = epoch, device = device)
        test_loss, test_mIoU = test_step(model, test_loader, criterion, device)

        # Print result
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train mIoU: {train_mIoU:.4f}, Test Loss: {test_loss:.4f}, Test mIoU: {test_mIoU:.4f}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_mIoU"].append(train_mIoU)
        results["test_loss"].append(test_loss)
        results["test_mIoU"].append(test_mIoU)
    return results

def train_with_writer(model: nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, experiment_name = 'tennis', model_name = 'TrackNetV2', criterion = nn.CrossEntropyLoss(), epochs: int = 5, device: str = 'cuda') -> dict[str, list[float]]:
    """
    Trains a PyTorch model.
    Args:
        module (torch.nn.Module): A PyTorch module.
        train_loader (torch.utils.data.DataLoader): A PyTorch DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): A PyTorch DataLoader for the testing dataset.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer.
        criterion (torch.nn.Module): A PyTorch loss function.
        epochs (int): Number of epochs to train the model for.
        device (str): Device to train the model on.
        writer (torch.utils.tensorboard.SummaryWriter): A PyTorch SummaryWriter to log metrics to TensorBoard.

    Returns:
        dict[str, list[float]]: A dictionary containing the training and testing loss and accuracy for each epoch.
    """
    writer = create_writer(experiment_name = experiment_name, model_name = model_name)   
    writer.add_graph(model, next(iter(train_loader))[0].to(device))
    results = {'train_loss': [], 'train_mIoU': [], 'test_loss': [], 'test_mIoU': []}
    evaluator = SegmentationMetric(2)
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        train_loss, train_mIoU = train_step(model = model, train_loader = train_loader, optimizer = optimizer, criterion = criterion, evaluator = evaluator, epoch = epoch, device = device)
        test_loss, test_mIoU = test_step(model = model, test_loader = test_loader, criterion = criterion, evaluator = evaluator, device = device)

        # Print result
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train mIoU: {train_mIoU:.4f}, Test Loss: {test_loss:.4f}, Test mIoU: {test_mIoU:.4f}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_mIoU"].append(train_mIoU)
        results["test_loss"].append(test_loss)
        results["test_mIoU"].append(test_mIoU)

        # Log metrics to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("mIoU/train", train_mIoU, epoch)
            writer.add_scalar("mIoU/test", test_mIoU, epoch)
        # Save model
        save_model(model, experiment_name, model_name + f'_epoch_{epoch}.pth')
    return results