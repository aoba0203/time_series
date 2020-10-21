import os
from pathlib import Path

def makeDirs(_path):
  if not os.path.exists(_path):
    os.makedirs(_path)
  return _path

def getProjectRootPath():
  root_path = Path(os.path.abspath('definitions.py'))  
  return root_path.parents[0]

def getTrainedModelPath(_model_name, _prefix):
  root_path = Path(getProjectRootPath())
  trained_path = os.path.join(root_path, 'trained_models')
  model_path = os.path.join(trained_path, _model_name)
  prefix_path = os.path.join(model_path, _prefix)
  return makeDirs(prefix_path)