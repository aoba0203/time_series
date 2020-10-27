import torch
from torch.nn import functional as F

NAME_LOSS_MAPE = 'MAPE'
NAME_LOSS_MSE = 'MSE'
NAME_LOSS_MASE = 'MASE' # Mean Absolute Scale Error
NAME_LOSS_PA = 'PA' # Predict Accuracy 1 - ((P - Y) / (P + Y))

def calcLossMAPE(_pred, _y):
  return torch.mean(torch.abs(_pred - _y) / torch.abs(_pred))

def calcLossMSE(_pred, _y):
  return F.mse_loss(_pred, _y)

def calcLossMASE(_pred, _y):
  count_ = _pred.shape[0]
  diff_ = torch.abs(_y[1:] - _y[:-1]).sum() / (count_-1)
  errors_ = torch.abs(_y - _pred)
  return errors_.mean() / diff_

def calcLossPA(_pred, _y):
  return torch.mean(1 - (torch.abs(_pred - _y) / torch.abs(_pred + _y)))

