import pickle
import os
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from .callbacks import EpochCounter, HistoryLogger
from .utils import merge_dicts_with_only_lists_as_values

class ResumableModel(object):
  """Save and overwrite a model every 'save_every_epochs' epochs to 'to_path',
  preserving the number of epochs and the history dict over multiple interrupted
  executions.

  If to_path is mymodel.h5, then there will be mymodel_epoch_num.pkl and 
  mymodel_history.pkl in the same directory as mymodel.h5, which hold backups for 
  the epoch counter and the history dict, respectively.

  Args:
    save_every_epochs (int): How often to save the model and the accompanying 
      parameters.
    to_path (str): A path to a model destination with the .h5 extension, which is 
      where model weights will be saved.

  Returns: A Keras History.history dictionary of the entire training process.
  """
  def __init__(self, model, custom_objects=None, save_every_epochs=10, to_path="model.h5"):
    
    assert save_every_epochs > 0

    self.model = model
    self.save_every_epochs = save_every_epochs
    self.custom_objects = custom_objects
    self.to_path = to_path
    self.prefix = os.path.splitext(to_path)[0]
    self.epoch_num_file = self.prefix + "_epoch_num.pkl"
    self.history_file = self.prefix + "_history.pkl"

    # recover latest epoch
    self.initial_epoch = self.get_epoch_num()
    # recover history
    self.history = self.get_history()
    # recover model from path
    if os.path.exists(to_path):
      self.model = load_model(to_path, custom_objects=self.custom_objects)
      logger = logging.getLogger()
      logger.info(f"Recovered model from {to_path} at epoch {self.initial_epoch}.")

  def _load_pickle(self, filePath, default_value):
    return pickle.load(open(filePath, 'rb')) if os.path.exists(filePath) else default_value
  
  def get_epoch_num(self):
    return self._load_pickle(self.epoch_num_file, 0)
  
  def get_history(self):
    return self._load_pickle(self.history_file, {})

  def _make_fit_args(self, *args, **kwargs):
    assert not 'initial_epoch' in kwargs
    # add callbacks for periodic checkpointing
    if 'callbacks' not in kwargs:
      kwargs['callbacks'] = []
    kwargs['callbacks'].append(HistoryLogger(period=self.save_every_epochs, history_path=self.history_file, recovered_history=self.history))
    kwargs['callbacks'].append(ModelCheckpoint(self.to_path, verbose=True, period=self.save_every_epochs))
    kwargs['callbacks'].append(EpochCounter(period=self.save_every_epochs, counter_path=self.epoch_num_file))
    # Warn user if the training is already complete.
    if 'epochs' in kwargs and self.initial_epoch >= kwargs['epochs']:
      epochs = kwargs['epochs']
      logger = logging.getLogger()
      logger.warning(f'You want to train for {epochs} epochs but {self.initial_epoch} epochs already completed; nothing to do.')
    return args, kwargs
  
  def _perform_final_save(self, remaining_history, epoch):
    # Combine histories and save
    combined_history = merge_dicts_with_only_lists_as_values([self.history, remaining_history.history])
    pickle.dump(combined_history, open(self.history_file, "wb"))
    # Dump last last epoch
    pickle.dump(epoch, open(self.epoch_num_file, "wb"))
    # Save model
    self.model.save(self.to_path)
    return combined_history
  
  def fit(self, *args, **kwargs):
    args, kwargs = self._make_fit_args(*args, **kwargs)
    remaining_history = self.model.fit(initial_epoch=self.initial_epoch, *args, **kwargs)
    combined_history = self._perform_final_save(remaining_history, epoch=kwargs['epochs'])
    return combined_history
  
  def fit_generator(self, *args, **kwargs):
    args, kwargs = self._make_fit_args(*args, **kwargs)
    remaining_history = self.model.fit_generator(initial_epoch=self.initial_epoch, *args, **kwargs)
    combined_history = self._perform_final_save(remaining_history, epoch=kwargs['epochs'])
    return combined_history
