from tensorflow import keras 
import pickle

from .utils import merge_dicts_with_only_lists_as_values

class EpochCounter(keras.callbacks.Callback):
  def __init__(self, period, counter_path):
    self.period = period
    self.counter_path = counter_path
    self.clock = 0
    super(EpochCounter, self).__init__()
  def on_epoch_begin(self, epoch, logs=None):
    # save epoch number to disk
    self.clock += 1
    if self.clock % self.period == 1 or self.period == 1:
      pickle.dump(epoch, open(self.counter_path, "wb"))

class HistoryLogger(keras.callbacks.Callback):
  def __init__(self, period, history_path, recovered_history):
    self.period = period
    self.recovered_history = recovered_history 
    self.history_path = history_path
    self.clock = 0
    super(HistoryLogger, self).__init__()
  def on_epoch_begin(self, epoch, logs=None):
    self.clock += 1
    if self.clock % self.period == 1 or self.period == 1:
      combined_history = merge_dicts_with_only_lists_as_values([self.recovered_history, self.model.history.history])
      pickle.dump(combined_history, open(self.history_path, "wb"))