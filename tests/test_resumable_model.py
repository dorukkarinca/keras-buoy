# -*- coding: utf-8 -*-

import pytest
from keras_buoy.models import ResumableModel

__author__ = "Doruk Karınca"
__copyright__ = "Doruk Karınca"
__license__ = "mit"

import os
import pickle
import numpy as np
from tensorflow import keras 
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import shutil

def build_model_and_data(custom_loss=None):
  # Build model and random data
  model = keras.Sequential()
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(128, activation='relu'))
  model.compile(loss=('binary_crossentropy' if not custom_loss else custom_loss),
              optimizer='adam')
  x_train = np.random.rand(1000, 16)
  y_train = np.random.rand(1000,)
  return model, x_train, y_train

def test_negative_epochs():
  # Cannot have negative epochs
  model, _, _ = build_model_and_data()
  with pytest.raises(AssertionError):
    resumable_model = ResumableModel(model, save_every_epochs=-4, to_path='/tmp/nonexistentmodel.h5')

def test_initial_epoch_set():
  model, x_train, y_train = build_model_and_data()
  # Cannot set initial_epoch
  with pytest.raises(AssertionError):
    resumable_model = ResumableModel(model, save_every_epochs=4, to_path='/tmp/nonexistentmodel.h5')
    history = resumable_model.fit(x=x_train, y=y_train, validation_split=0.1, batch_size=256, verbose=2, epochs=12, initial_epoch=1)

def single_run(num_epochs, save_every_epochs, custom_loss=None, to_path='/tmp/mymodel.h5'):
  model, x_train, y_train = build_model_and_data(custom_loss=custom_loss)
  prefix = os.path.splitext(to_path)[0]
  epoch_num_file = prefix + "_epoch_num.pkl"
  history_file = prefix + "_history.pkl"
  filesCreated = [to_path, epoch_num_file, history_file]
  # test resumable model with random data
  for path in filesCreated:
    if os.path.exists(path) and os.path.isfile(path):
      os.remove(path)
    elif os.path.exists(path) and os.path.isdir(path):
      shutil.rmtree(path)
  resumable_model = ResumableModel(model, save_every_epochs=save_every_epochs, to_path=to_path, custom_objects={'custom_loss': custom_loss})
  history = resumable_model.fit(x=x_train, y=y_train, validation_split=0.1, batch_size=256, verbose=2, epochs=num_epochs)

  assert history != {} 
  # Check the saved history is identical to the one returned
  assert history == pickle.load(open(history_file, 'rb'))
  assert isinstance(history, dict) 
  for k, v in history.items():
    assert len(v) == num_epochs
  
  for path in filesCreated:
    assert os.path.exists(path)
    if os.path.isfile(path):
      os.remove(path)
    else:
      shutil.rmtree(path)

def test_resumable_model_single_run_with_overlapping_multiple():
  single_run(num_epochs=12, save_every_epochs=4)

def test_resumable_model_single_run_with_remainder_epoch():
  single_run(num_epochs=13, save_every_epochs=4)

def test_resumable_model_single_run_with_save_every_epoch_1():
  single_run(num_epochs=3, save_every_epochs=1)

def test_resumable_model_single_run_with_single_epoch():
  single_run(num_epochs=1, save_every_epochs=1)

def test_resumable_model_single_run_with_custom_loss():
  def custom_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)        
  single_run(num_epochs=3, save_every_epochs=1, custom_loss=custom_loss)

def test_resumable_model_single_run_with_tensorflow_savedmodel():
  single_run(num_epochs=3, save_every_epochs=1, to_path='/tmp/mymodel')

def test_resumable_model_interrupted_run():
  model, x_train, y_train = build_model_and_data()
  filesCreated = ['/tmp/mymodel.h5', '/tmp/mymodel_epoch_num.pkl', '/tmp/mymodel_history.pkl']

  # test resumable model with random data
  NUM_EPOCHS = 12
  for filePath in filesCreated:
    if os.path.exists(filePath):
      os.remove(filePath)
  resumable_model = ResumableModel(model, save_every_epochs=4, to_path='/tmp/mymodel.h5')
  history = resumable_model.fit(x=x_train, y=y_train, validation_split=0.1, batch_size=256, verbose=2, epochs=NUM_EPOCHS)

  assert history != {} 
  assert isinstance(history, dict) 
  for k, v in history.items():
    assert len(v) == NUM_EPOCHS
  for filePath in filesCreated:
    assert os.path.exists(filePath)

  # add more epochs and retrain (it should open the old history dicts and keep adding)
  NUM_MORE_EPOCHS = 20
  NUM_NEW_EPOCHS = NUM_EPOCHS + NUM_MORE_EPOCHS
  model, _, _ = build_model_and_data() # refresh model so weights are lost
  resumable_model = ResumableModel(model, save_every_epochs=4, to_path='/tmp/mymodel.h5') # recover model
  new_history = resumable_model.fit(x=x_train, y=y_train, validation_split=0.1, batch_size=256, verbose=2, epochs=NUM_NEW_EPOCHS)

  # Ensure the new history dict is a longer version of the older history dict
  old_history = history
  for k, v in new_history.items():
    assert len(v) == NUM_NEW_EPOCHS
    assert k in old_history
    assert old_history[k] == v[:len(old_history[k])]

  for filePath in filesCreated:
    os.remove(filePath)