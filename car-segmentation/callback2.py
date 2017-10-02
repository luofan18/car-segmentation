from keras import __version__ as kv
import numpy as np
from keras.callbacks import Callback
import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
from math import ceil
from itchat.content import TEXT
import os
from os import system
import re
import traceback

class BatchHistory(Callback):
    def __init__(self, save_to, save_figure=True, save_data=False, ):
        self.save_to = save_to
        self.save_figure = save_figure
        self.save_data = save_data
        self.epoch = 0
    
    def on_train_begin(self, logs=None):
        self.batch = []
        self.batch_history = {}
        
        self.epochs = []
        self.epoch_history = {}
        
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        
        epoch = self.epoch
        steps_per_epoch = self.params['steps']
        # The first batch starts with 1
        batch = steps_per_epoch * epoch + batch + 1
        
        self.batch.append(batch)
        for k in self.params['metrics']:
            if k in logs:
                self.batch_history.setdefault(k, []).append(logs[k])
                
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
    
    def on_epoch_end(self, epoch, logs=None):
        save_to = os.path.join(self.save_to,
                               'batch_history_epoch_' + str(epoch))
        batch = self.batch
        
        epochs_save_to = os.path.join(self.save_to, 
                                      'epoch_history')
        # Retrieve the validation data
        self.epochs.append(epoch)
        for k in logs:
            if k.startswith('val_'):
                self.epoch_history.setdefault(k, []).append(logs[k])
        
        if self.save_data:
            np.save(save_to + '.npy', self.batch_history)
            np.save(epochs_save_to + '.npy', self.epochs)
        if self.save_figure:
            # save batch figure
            row_num = len(self.batch_history)
            fig, axes = plt.subplots(row_num, figsize=(10, 6 * row_num))
            i = 0
            for metric, data in self.batch_history.items():
                curr_ax = axes[i]
                curr_ax.plot(batch, data, label=metric)
                curr_ax.set_title(metric + ' in batch')
                curr_ax.set_xlabel('batch')
                curr_ax.set_ylabel(metric)
                i = i + 1
            fig.savefig(save_to + '.jpg')
            plt.close(fig)
            
            # Draw the figure
            fig, axes = plt.subplots(row_num, figsize=(10, 6 * row_num))   
            i = 0                          
            for metric, data in self.epoch_history.items():
                curr_ax = axes[i]
                curr_ax.plot(self.epochs, data, label=metric)
                curr_ax.set_title(metric + ' in epoch')
                curr_ax.set_xlabel('epoch')
                curr_ax.set_ylabel(metric)
                i = i + 1
            fig.savefig(epochs_save_to + '.jpg')
            plt.close(fig)
