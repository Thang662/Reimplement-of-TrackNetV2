import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = None, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        input = input.float()
        target = target.float()
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction = 'none')
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    
class FocalBCELoss(nn.Module):
    def __init__(self, gamma = 2, alpha = None, reduction = 'mean'):
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        input = input.float()
        target = target.float()
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction = 'none')
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss + ce_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.float()
        targets = targets.float()
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.float()
        targets = targets.float()
        
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
class FocalDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalDiceLoss, self).__init__()

    def forward(self, inputs, targets, alpha=None, gamma=2, smooth=1):
        inputs = inputs.float()
        targets = targets.float()
        
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE_EXP = torch.exp(-BCE)
        focal_loss = (1-BCE_EXP)**gamma * BCE
        if alpha is not None:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        FocalDice = focal_loss.mean() + dice_loss
        
        return FocalDice
    
class FocalDiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalDiceBCELoss, self).__init__()

    def forward(self, inputs, targets, alpha=None, gamma=2, smooth=1):
        inputs = inputs.float()
        targets = targets.float()
        
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE_EXP = torch.exp(-BCE)
        focal_loss = (1-BCE_EXP)**gamma * BCE
        if alpha is not None:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        FocalDiceBCE = focal_loss.mean() + dice_loss + BCE.mean()
        
        return FocalDiceBCE

class FocalLossv2(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLossv2, self).__init__()

    def forward(self, inputs, targets, alpha=0, gamma=2, smooth=1):
        inputs = inputs.float()
        targets = targets.float()
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = (1-BCE_EXP)**gamma * BCE
        if alpha is not None:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
                       
        return focal_loss.mean()

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(input.size(0), input.size(1), -1)
#         target = target.transpose(1, 2)
#         target = target.contiguous().view(-1, target.size(2))

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()