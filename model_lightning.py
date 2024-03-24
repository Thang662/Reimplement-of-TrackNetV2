from typing import Any
import lightning as L
from model import *
from loss import *
from lightning.pytorch.utilities.model_summary import ModelSummary
from torchvision.utils import make_grid
import wandb

class LitTrackNetV2(L.LightningModule):
    def __init__(self, frame_in, frame_out, norm_layer = nn.BatchNorm2d, optimizer = torch.optim.Adam, loss_fn = FocalLoss(), weight_init = None, log_image_every_n_steps = 10, lr = 0.001):
        super().__init__()
        self.example_input_array = torch.Tensor(2, frame_in, 512, 288)
        self.net = TrackNetV2(in_channels = frame_in, out_channels = frame_out, norm_layer = norm_layer)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.weight_init = weight_init
        if self.weight_init:
            self.net.apply(weight_init)
        self.training_step_intersections = []
        self.training_step_unions = []
        self.validation_step_intersections = []
        self.validation_step_unions = []
        self.log_image_every_n_steps = log_image_every_n_steps
        self.lr = lr


    def on_train_start(self):
        tensorboard_logger = self.loggers[0].experiment
        wandb_logger = self.loggers[1].experiment
        tensorboard_logger.add_graph(self.net, self.example_input_array.to(self.device))
        wandb_logger.watch(self.net, log = "all", log_graph = True)
    
    def training_step(self, batch, batch_idx):
        imgs, heatmaps, annos, annos_transformed = batch

        logits = self.net(imgs)

        loss = self.loss_fn(logits, heatmaps)

        preds = (torch.sigmoid(logits) > 0.5).float()
        intersection = torch.sum(preds * heatmaps)
        union = torch.logical_or(preds, heatmaps)
        iou = torch.sum(intersection) / torch.sum(union)
        self.training_step_intersections.append(intersection)
        self.training_step_unions.append(union)

        if batch_idx % self.log_image_every_n_steps == 0:
            tensorboard_logger = self.loggers[0].experiment
            wandb_logger = self.loggers[1].experiment
            grid = make_grid([preds[0][:1], heatmaps[0][:1], preds[0][1:2], heatmaps[0][1:2], preds[0][2:3], heatmaps[0][2:3]], nrow = 2, value_range = (0, 1), pad_value = 1)
            tensorboard_logger.add_image(f'Comparison/{self.current_epoch + 1}', grid, global_step = batch_idx)
            wandb_logger.log({f"Comparison_{self.current_epoch + 1}": [wandb.Image(grid, caption = f"Epoch {self.current_epoch + 1} Iteration {batch_idx}")]})


        self.log('train_loss', loss, prog_bar = True, logger = True, on_step = True, on_epoch = True)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # self.log('train_loss', outputs['loss'], prog_bar = True, logger = True, on_step = True, on_epoch = True)
        # self.log('train_iou', outputs['iou'], prog_bar = True, logger = True, on_step = True, on_epoch = True)
        pass

    def on_train_epoch_end(self):
        epoch_intersection = torch.stack(self.training_step_intersections).sum()
        epoch_union = torch.cat(self.training_step_unions).sum()
        epoch_miou = epoch_intersection / epoch_union
        self.log('train_mIoU', epoch_miou, prog_bar = True, logger = True, on_step = False, on_epoch = True)
        self.training_step_intersections.clear()
        self.training_step_unions.clear()
    
    def validation_step(self, batch, batch_idx):
        imgs, heatmaps, annos, annos_transformed = batch

        logits = self.net(imgs)

        loss = self.loss_fn(logits, heatmaps)

        preds = (torch.sigmoid(logits) > 0.5).float()
        intersection = torch.sum(preds * heatmaps)
        union = torch.logical_or(preds, heatmaps)
        self.validation_step_intersections.append(intersection)
        self.validation_step_unions.append(union)

        if batch_idx % self.log_image_every_n_steps == 0:
            grid = make_grid([preds[0][:1], heatmaps[0][:1], preds[0][1:2], heatmaps[0][1:2], preds[0][2:3], heatmaps[0][2:3]], nrow = 2, value_range = (0, 1), pad_value = 1)
            self.logger.experiment.add_image(f'Comparison/{self.current_epoch}', grid, global_step = batch_idx)

        self.log('val_loss', loss, prog_bar = True, logger = True, on_step = True, on_epoch = True)
        return loss
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        # self.log('val_loss', outputs, prog_bar = True, logger = True, on_step = True, on_epoch = True)
        pass
    
    def on_validation_epoch_end(self):
        epoch_intersection = torch.stack(self.validation_step_intersections).sum()
        epoch_union = torch.cat(self.validation_step_unions).sum()
        epoch_miou = epoch_intersection / epoch_union
        self.log('val_mIoU', epoch_miou, prog_bar = True, logger = True, on_step = False, on_epoch = True)
        self.validation_step_intersections.clear()
        self.validation_step_unions.clear()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr = self.lr)
    
    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = LitTrackNetV2(frame_in = 9, frame_out = 3)
    summary = ModelSummary(model, max_depth = -1)
    print(summary)
        