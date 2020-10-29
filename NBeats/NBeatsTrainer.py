from os import name
from matplotlib.pyplot import subplots
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch._C import dtype
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.operators import NAME_LOSS_MAPE, NAME_LOSS_MSE, NAME_LOSS_MASE, NAME_LOSS_PA
from NBeats.NBEATS import NBeatsNet
import definitions
from utils import operators
from NBeats import NBeatsDatasetMaker

CHECKPOINT_NAME_NET_STATE = 'dict_net_state'
CHECKPOINT_NAME_OPTIMZER_STATE = 'dict_optimizer_state'
CHECKPOINT_NAME_STEP = 'step'

NAME_ENSEMBLE_SET_BIG = 'ensemble_big'
NAME_ENSEMBLE_SET_MIDDLE = 'ensemble_middle'
NAME_ENSEMBLE_SET_SMALL = 'ensemble_small'

NAME_ENSEMBLE_EPOCH = 'ensemble_epoch'
NAME_ENSEMBLE_LOSS = 'ensemble_loss'
NAME_ENSEMBLE_BACKCAST = 'ensemble_BACKCAST'

NAME_EPOCH_5K = '5k'
NAME_EPOCH_10K = '10k'
NAME_EPOCH_15K = '15k'

NAME_BACKCAST_2H = '2h'
NAME_BACKCAST_3H = '3h'
NAME_BACKCAST_4H = '4h'
NAME_BACKCAST_5H = '5h'
NAME_BACKCAST_6H = '6h'
NAME_BACKCAST_7H = '7h'
NAME_BACKCAST_8H = '8h'
NAME_BACKCAST_9H = '9h'
NAME_BACKCAST_10H = '10h'

LIST_ENSEMBLE_BIG_EPOCH = [ NAME_EPOCH_5K, NAME_EPOCH_10K, NAME_EPOCH_15K ]
LIST_ENSEMBLE_MIDDLE_EPOCH = [ NAME_EPOCH_5K, NAME_EPOCH_10K ]
LIST_ENSEMBLE_SMALL_EPOCH = [ NAME_EPOCH_5K]

LIST_ENSEMBLE_BIG_LOSS = [ NAME_LOSS_MAPE, NAME_LOSS_MSE, NAME_LOSS_MASE, NAME_LOSS_PA ]
LIST_ENSEMBLE_MIDDLE_LOSS = [ NAME_LOSS_MAPE, NAME_LOSS_MSE, NAME_LOSS_MASE ]
LIST_ENSEMBLE_SMALL_LOSS = [ NAME_LOSS_MAPE, NAME_LOSS_MASE ]

LIST_ENSEMBLE_BIG_BACKCAST = [ NAME_BACKCAST_2H, NAME_BACKCAST_3H, NAME_BACKCAST_4H, NAME_BACKCAST_5H, NAME_BACKCAST_6H, NAME_BACKCAST_7H, NAME_BACKCAST_8H, NAME_BACKCAST_9H, NAME_BACKCAST_10H ]
LIST_ENSEMBLE_MIDDLE_BACKCAST = [ NAME_BACKCAST_2H, NAME_BACKCAST_3H, NAME_BACKCAST_4H, NAME_BACKCAST_5H, NAME_BACKCAST_6H, NAME_BACKCAST_7H ]
LIST_ENSEMBLE_SMALL_BACKCAST = [ NAME_BACKCAST_2H, NAME_BACKCAST_3H, NAME_BACKCAST_4H, NAME_BACKCAST_5H ]

DICT_EPOCH = { NAME_EPOCH_5K:5000, NAME_EPOCH_10K:10000, NAME_EPOCH_15K:15000 }
DICT_BACKCAST = { NAME_BACKCAST_2H:2, NAME_BACKCAST_3H:3, NAME_BACKCAST_4H:4, NAME_BACKCAST_5H:5, NAME_BACKCAST_6H:6, NAME_BACKCAST_7H:7, NAME_BACKCAST_8H:8, NAME_BACKCAST_9H:9, NAME_BACKCAST_10H:10 }
DICT_LOSS_FUNCTION = {NAME_LOSS_MAPE: operators.calcLossMAPE, NAME_LOSS_MSE: operators.calcLossMSE, NAME_LOSS_MASE: operators.calcLossMASE, NAME_LOSS_PA:operators.calcLossPA}
DICT_ENSEMBLE = {
  NAME_ENSEMBLE_SET_BIG:{
    NAME_ENSEMBLE_EPOCH: LIST_ENSEMBLE_BIG_EPOCH,
    NAME_ENSEMBLE_LOSS: LIST_ENSEMBLE_BIG_LOSS,
    NAME_ENSEMBLE_BACKCAST: LIST_ENSEMBLE_BIG_BACKCAST,
  },
  NAME_ENSEMBLE_SET_MIDDLE:{
    NAME_ENSEMBLE_EPOCH: LIST_ENSEMBLE_MIDDLE_EPOCH,
    NAME_ENSEMBLE_LOSS: LIST_ENSEMBLE_MIDDLE_LOSS,
    NAME_ENSEMBLE_BACKCAST: LIST_ENSEMBLE_MIDDLE_BACKCAST,
  },
  NAME_ENSEMBLE_SET_SMALL:{
    NAME_ENSEMBLE_EPOCH: LIST_ENSEMBLE_SMALL_EPOCH,
    NAME_ENSEMBLE_LOSS: LIST_ENSEMBLE_SMALL_LOSS,
    NAME_ENSEMBLE_BACKCAST: LIST_ENSEMBLE_SMALL_BACKCAST,
  },
}

class NBeatsTrainer:
  tag = 'NBeatsTrainer'
  # prefix = tag
  name_loss = NAME_LOSS_MAPE
  name_epoch = NAME_EPOCH_5K
  name_backcast = NAME_BACKCAST_2H  

  def __init__(self, _prefix):
    self.prefix = _prefix
    self.global_loss = 100000
    self.model_path = definitions.getTrainedModelPath(self.tag, self.prefix)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def makePlotScatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)

  def printPlot(self, _x, _y, _forecast):    
    __subplots = [221, 222, 223, 224]
    plt.figure(1)
    for __plot_id, __idx in enumerate(np.random.choice(range(len(_x)), size=4, replace=False)):
      __plot_x, __plot_y, __plot_forecast = _x[__idx].reshape([-1, 1]).cpu().detach().numpy(), _y[__idx].cpu().detach().numpy(), _forecast[__idx].cpu().detach().numpy()
      __len_backcast = len(__plot_x)
      __len_forecast = len(__plot_forecast)
      plt.subplot(__subplots[__plot_id])
      plt.grid()
      plt.plot(np.arange(0, __len_backcast), __plot_x, color='b')
      plt.plot(np.arange(__len_backcast-1, __len_backcast + __len_forecast-1), __plot_y, color='g')
      plt.plot(np.arange(__len_backcast-1, __len_backcast + __len_forecast-1), __plot_forecast, color='r')
#       self.makePlotScatter(np.arange(0, __len_backcast), __plot_x, color='b')
#       self.makePlotScatter(np.arange(__len_backcast, __len_backcast + __len_forecast), __plot_y, color='g')
#       self.makePlotScatter(np.arange(__len_backcast, __len_backcast + __len_forecast), __plot_forecast, color='r')
    plt.show()

  def loadFromFile(self, _path_model_file, _net, _optimizer):
    __checkpoints = torch.load(_path_model_file)
    _net.load_state_dict(__checkpoints[CHECKPOINT_NAME_NET_STATE])
    _optimizer.load_state_dict(__checkpoints[CHECKPOINT_NAME_OPTIMZER_STATE])
    __epoch_step = __checkpoints[CHECKPOINT_NAME_STEP]
    return __epoch_step

  def load(self, _net, _optimizer, _name_backcast, _name_epoch, _name_loss):    
    __model_file_name = self.getModelName(_name_backcast, _name_epoch, _name_loss)
    __model_file_path = os.path.join(self.model_path, __model_file_name)
    if os.path.exists(__model_file_path):
      # __checkpoints = torch.load(__model_file_path)
      # _net.load_state_dict(__checkpoints[CHECKPOINT_NAME_NET_STATE])
      # _optimizer.load_state_dict(__checkpoints[CHECKPOINT_NAME_OPTIMZER_STATE])
      # __epoch_step = __checkpoints[CHECKPOINT_NAME_STEP]
      __epoch_step = self.loadFromFile(__model_file_path, _net, _optimizer)
      # print(f'load checkpoint from {__model_file_path}.')
      return __epoch_step
    return 0
  
  def save(self, _net, _optimizer, _step, _name_backcast, _name_epoch, _name_loss, _best=None):
    __model_file_name = self.getModelName(_name_backcast, _name_epoch, _name_loss)
    if _best:
      __model_file_name = 'best_' + __model_file_name
    __model_file_path = os.path.join(self.model_path, __model_file_name)
    torch.save({
      CHECKPOINT_NAME_STEP: _step,
      CHECKPOINT_NAME_NET_STATE: _net.state_dict(),
      CHECKPOINT_NAME_OPTIMZER_STATE: _optimizer.state_dict(),
    }, __model_file_path)
  
  def getModelName(self, _name_backcast, _name_epoch, _name_loss):
    return self.prefix + '_' + _name_epoch + '_' + _name_backcast + '_' + _name_loss + '_' + str(self.device) + '.ckpt'
  
  def getNet(self, _len_forecast, _len_backcast):    
    __cnt_seasonality_stack = 1
    __cnt_trend_stack = 1
    __cnt_generic_stack = 30    
    __list_stack = []
    __list_block_cnt = []
    __list_hidden_layer_units = []
    __list_thetas_dims = []
    for __ in range(__cnt_seasonality_stack):
      __list_stack.append(NBeatsNet.SEASONALITY_BLOCK)
      __list_hidden_layer_units.append(2048)
      __list_block_cnt.append(3)
      __list_thetas_dims.append(20)
    for __ in range(__cnt_trend_stack):
      __list_stack.append(NBeatsNet.TREND_BLOCK)
      __list_hidden_layer_units.append(256)
      __list_block_cnt.append(4)
      __list_thetas_dims.append(4)
    for __ in range(__cnt_generic_stack):
      __list_stack.append(NBeatsNet.GENERIC_BLOCK)
      __list_hidden_layer_units.append(512)
      __list_block_cnt.append(1)
      __list_thetas_dims.append(20)
    __net = NBeatsNet(device=self.device,
      stack_types=__list_stack,
      forecast_length=_len_forecast,
      backcast_length=_len_backcast,
      hidden_layer_units=__list_hidden_layer_units,
      nb_blocks_per_stack=__list_block_cnt,
      thetas_dims=__list_thetas_dims,
      share_weights_in_stack=False,
      )
    __optimizer = optim.Adam(__net.parameters())
    scheduler = lr_scheduler.ExponentialLR(__optimizer, gamma= 0.99)    
    __net.to(self.device)
    return __net, __optimizer, scheduler
  
  def evaluation(self, _x, _y, _net, _optimizer, _name_backcast=NAME_BACKCAST_3H, _name_epoch=NAME_EPOCH_5K, _name_loss=NAME_LOSS_MAPE, _print_plot=False):
    with torch.no_grad():
      __model_name = self.getModelName(_name_backcast, _name_epoch, _name_loss)
      __loss_function = DICT_LOSS_FUNCTION[_name_loss]
      __train_epoch = self.load(_net, _optimizer, _name_backcast, _name_epoch, _name_loss)
      _net.eval()
      __backcast, __forecast = _net(_x)
      __loss = __loss_function(__forecast, _y)
      if _print_plot:
        self.printPlot(_x, _y, __forecast)
      # print(f'Evaluation - Name = {str(__model_name)}, loss = {__loss.item():.6f}')
      if __loss < self.global_loss:
        print(f'Refesh Best Model - Name = {str(__model_name)}, loss = {self.global_loss:.6f} -> {__loss.item():.6f}')
        self.global_loss = __loss.item()
        self.save(_net, _optimizer, 0, _name_backcast, _name_epoch, _name_loss, 'best')        

  def train_epoch(self, _epoch, _x, _y, _net, _optimizer, _name_backcast=NAME_BACKCAST_3H, _name_epoch=NAME_EPOCH_5K, _name_loss=NAME_LOSS_MAPE, _batch_size=256, _print_epoch=100, _shuffle=True, _save=True):
    __train_step = self.load(_net, _optimizer, _name_backcast, _name_epoch, _name_loss)
    __loss_function = DICT_LOSS_FUNCTION[_name_loss]
    __datasets = TensorDataset(_x, _y)
    __data_loader = DataLoader(__datasets, batch_size=_batch_size, shuffle=_shuffle)    
    __loss = torch.tensor(0.).to(self.device)
    _net.train()
    for __x, __y in __data_loader:
      _optimizer.zero_grad()
      __backcast, __forecast = _net(__x)
      __loss = __loss_function(__forecast, __y)
      __loss.backward()
      _optimizer.step()
      __train_step += 1      
    if __train_step % _print_epoch == 0:
      print(f'Train - Step = {str(__train_step).zfill(6)}, loss({_name_loss}) = {__loss.item():.6f}')
      # del __loss, __forecast, __x, __y
    if _save:
      with torch.no_grad():
        self.save(_net, _optimizer, __train_step, _name_backcast, _name_epoch, _name_loss)        
    return __train_step

  def train(self, _data_train, _data_eval, _list_data_name, _len_forecast, _batch_size=1024, _name_ensemble_set=NAME_ENSEMBLE_SET_SMALL):
    # self.model_path = definitions.getTrainedModelPath(self.tag, _name_ensemble_set)
    self.global_loss = 100000
    __list_epoch = DICT_ENSEMBLE[_name_ensemble_set][NAME_ENSEMBLE_EPOCH]
    __list_loss = DICT_ENSEMBLE[_name_ensemble_set][NAME_ENSEMBLE_LOSS]
    __list_backcast = DICT_ENSEMBLE[_name_ensemble_set][NAME_ENSEMBLE_BACKCAST]
  
    for __name_epoch in __list_epoch:  
      for __name_backcast in __list_backcast:
        __len_backcast = _len_forecast * DICT_BACKCAST[__name_backcast]        
        __train_x, __train_y, __train_idx = NBeatsDatasetMaker.makeDataset(_data_train, __len_backcast, _len_forecast, _list_data_name, self.device)
        __eval_x, __eval_y, __eval_idx = NBeatsDatasetMaker.makeDataset(_data_eval, __len_backcast, _len_forecast, _list_data_name, self.device)        
        for __name_loss in __list_loss:
          __net, __optimizer, __scheduler = self.getNet(_len_forecast, __len_backcast)
          for __epoch in range(DICT_EPOCH[__name_epoch]):
            __train_step = self.train_epoch(__epoch, __train_x, __train_y, __net, __optimizer, __name_backcast, __name_epoch, __name_loss, _batch_size)
            __scheduler.step()
            if __train_step > DICT_EPOCH[__name_epoch]:
              break
          self.evaluation(__eval_x, __eval_y, __net, __optimizer, __name_backcast, __name_epoch, __name_loss)

  def train_one(self, _data_train, _data_eval, _list_data_name, _len_forecast, __name_backcast=NAME_BACKCAST_7H, __name_epoch=NAME_EPOCH_15K, __name_loss=NAME_LOSS_MSE, _batch_size=1024):
    self.global_loss = 100000
    __len_backcast = _len_forecast * DICT_BACKCAST[__name_backcast]        
    __train_x, __train_y, __train_idx = NBeatsDatasetMaker.makeDataset(_data_train, __len_backcast, _len_forecast, _list_data_name, self.device)
    __eval_x, __eval_y, __eval_idx= NBeatsDatasetMaker.makeDataset(_data_eval, __len_backcast, _len_forecast, _list_data_name, self.device)    
    __net, __optimizer, __scheduler = self.getNet(_len_forecast, __len_backcast)
    for __epoch in range(DICT_EPOCH[__name_epoch]):
      __train_step = self.train_epoch(__epoch, __train_x, __train_y, __net, __optimizer, __name_backcast, __name_epoch, __name_loss, _batch_size)
      __scheduler.step()
      if __train_step > DICT_EPOCH[__name_epoch]:
        break
      __print_plot = False
      if __epoch % 10 == 0:
        __print_plot = True
      self.evaluation(__eval_x, __eval_y, __net, __optimizer, __name_backcast, __name_epoch, __name_loss, _print_plot=__print_plot)

  def predict_ensemble(self, _x, _len_forecast, _name_ensemble_set=NAME_ENSEMBLE_SET_SMALL, _choice_function=np.median):    
    __path_model = definitions.getTrainedModelPath(self.tag, _name_ensemble_set)
    __list_predict = []
    for __path_model in glob.glob(__path_model + '/*'):
      __name_iterate = os.path.basename(__path_model).split('_')[1]
      __name_backcast = os.path.basename(__path_model).split('_')[2]
      __name_loss = os.path.basename(__path_model).split('_')[3]
      __forecast = self.predict(_x, _len_forecast, __name_backcast, __name_iterate, __name_loss)
      __list_predict.append(__forecast)
    return _choice_function(__list_predict, axis=0)

  def predict(self, _x, _len_forecast, _name_backcast=NAME_BACKCAST_3H, _name_epoch=NAME_EPOCH_5K, _name_loss=NAME_LOSS_MAPE):
    with torch.no_grad():
      __len_backcast = _len_forecast * DICT_BACKCAST[_name_backcast]
      _x = _x[:__len_backcast]
      __x_t = torch.tensor(_x, dtype=torch.float).to(self.device)      
      __net, __optimizer, __scheduler = self.getNet(_len_forecast, __len_backcast)      
      __train_epoch = self.load(__net, __optimizer, _name_backcast, _name_epoch, _name_loss)
      __net.eval()
      __backcast, __forecast = __net(__x_t)
      return __forecast.numpy()
    