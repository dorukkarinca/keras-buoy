==========
keras-buoy
==========

.. image:: https://travis-ci.com/dorukkarinca/keras-buoy.svg?branch=master
    :target: https://travis-ci.com/dorukkarinca/keras-buoy

Keras wrapper that autosaves and auto-recovers not just the model weights but also the last epoch number and training history metrics.

::

    pip install keras-buoy

::

    >>> resumableModel = ResumableModel(model, save_every_epochs=4, to_path='kerascheckpoint.h5')
    >>> history = resumableModel.fit(x = x_train, y = y_train, validation_split=0.1, batch_size = 256, verbose=2, epochs=15)

    Recovered model from kerascheckpoint.h5 at epoch 8.

    Epoch 9/15
    1125/1125 - 5s - loss: 0.4790 - top_k_categorical_accuracy: 0.9698 - val_loss: 1.1075 - val_top_k_categorical_accuracy: 0.9206
    Epoch 10/15
    1125/1125 - 5s - loss: 0.4758 - top_k_categorical_accuracy: 0.9701 - val_loss: 1.1119 - val_top_k_categorical_accuracy: 0.9214
    Epoch 11/15
    1125/1125 - 5s - loss: 0.4753 - top_k_categorical_accuracy: 0.9702 - val_loss: 1.1000 - val_top_k_categorical_accuracy: 0.9215
    Epoch 12/15
    ...

Description
===========

When training is interrupted due to a crash/accidental :code:`Ctrl+C` and you rerun the whole code, it recovers the model weights and the epoch counter to the last saved values. Then it resumes training as if nothing happened. At the end, the Keras History.history dictionaries are combined so that the training history looks like one single training run.

Example
=======

::
    
    from tensorflow import keras
    from keras_buoy.models import ResumableModel

    model = keras.Sequential()
    ...
    resumable_model = ResumableModel(model, save_every_epochs = 4, custom_objects=None, to_path='/path/to/save/model_weights.h5')
    history = resumable_model.fit(x = x_train, y = y_train, validation_split = 0.1, batch_size = 256, verbose = 2, epochs = 12)

Usage
=====
:code:`custom_objects (dict)` is passed into :code:`tf.keras.models.load_model(...)` so you can load your model with a custom loss for example.

:code:`save_every_epochs (int)` will save the model, history, and epoch counter every so often. In case of a crash, recovery will happen from the last saved epoch multiple.

:code:`to_path (str)` is where the model weights will be saved, and must have the :code:`.h5` extension.

:code:`resumable_model.fit(...)` is the same as Keras' :code:`model.fit(...)`.

It returns :code:`history` which is the history dict of the Keras History object. Note that it does not return the Keras.History object itself, just the dict.

If :code:`to_path` is :code:`mymodel.h5`, then there will be :code:`mymodel_epoch_num.pkl` and :code:`mymodel_history.pkl` in the same directory as :code:`mymodel.h5`, which hold backups for the epoch counter and the history dict, respectively.

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
