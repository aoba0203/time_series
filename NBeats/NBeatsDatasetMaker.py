import numpy as np
import pandas as pd
import torch

NAME_DATA_TYPE_3D = '3d'
NAME_DATA_TYPE_2D = '2d'

def __checkDataShape(_data):
  if (len(np.shape(_data))==2):
    return NAME_DATA_TYPE_3D
  elif (len(np.shape(_data))==1):
    return NAME_DATA_TYPE_2D
  else:
    raise Exception('Array should be of [n x 1] or [m x n x 1] shape')

def __makeTrainDataset(_data, _len_backcast, _len_forecast, _normalize):
  __list_x = []
  __list_y = []
  _data = np.array(np.reshape(_data, [-1,]), dtype='float')    
  _data = _data[~np.isnan(_data)]
  if (len(_data) < _len_backcast + _len_forecast):
    return __list_x, __list_y, 0.0001
  __num_max = np.max(_data) * 1.5
  if __num_max == 0:
    return __list_x, __list_y, 0.0001
  if _normalize:
    _data = _data / __num_max
  for idx in range(len(_data) - (_len_backcast + _len_forecast)):
    if sum(_data[idx : idx+_len_backcast] != 0) == 0:
      continue
    __list_x.append(_data[idx : idx+_len_backcast])
    __list_y.append(_data[idx+_len_backcast : idx + (_len_backcast + _len_forecast)])
  return __list_x, __list_y, __num_max

def makeDataset(_data, _len_backcast, _len_forecast, _list_data_name, _device, _normalize=True):
  __name_data_type = __checkDataShape(_data)
  __list_dataset_x = []
  __list_dataset_y = []
  __list_index = []
  __dict_max = {}
  if __name_data_type == NAME_DATA_TYPE_3D:
    for __idx, __data_sub in enumerate(_data):
      __list_x, __list_y, __num_max = __makeTrainDataset(__data_sub, _len_backcast, _len_forecast, _normalize)
      if len(__list_x) == 0:
        continue
      __list_dataset_x.extend(__list_x)
      __list_dataset_y.extend(__list_y)
      for _ in __list_x:
        __list_index.append(__idx)
      # __dict_max[_list_data_name[__idx]] = __num_max
  elif __name_data_type == NAME_DATA_TYPE_2D:
    __list_dataset_x, __list_dataset_y, __num_max = __makeTrainDataset(_data, _len_backcast, _len_forecast, _normalize)
    __dict_max[_list_data_name[0]] = __num_max
  return torch.tensor(__list_dataset_x, dtype=torch.float).to(_device), torch.tensor(__list_dataset_y, dtype=torch.float).to(_device), __list_index