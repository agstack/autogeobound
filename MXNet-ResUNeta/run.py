import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import visdom

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import image

import sys
sys.path.append('../../resuneta/src')
sys.path.append('../../resuneta/nn/loss')
sys.path.append('../../resuneta/models')
sys.path.append('../../')

from bound_dist import get_distance, get_boundary
from loss import Tanimoto_wth_dual
from resunet_d6_causal_mtskcolor_ddist import *
from resunet_d7_causal_mtskcolor_ddist import *

from datasets import *
from train import *
from evaluate import *


# hyperparameters
args = {}
lr = 0.001
epochs = 100
batch_size = 4
save_path = '../experiments/france/'
save_file_name = os.path.join(save_path, "model_2k.params")
n_filters = 16
n_classes = 1
args['batch_size'] = batch_size
env_name = 'france_nfilter'+str(n_filters)+'_lr'+str(lr)+'_bs'+str(batch_size)
vis = visdom.Visdom(port=8097, env=env_name)
args['visdom'] = vis
args['visdom_every'] = 10

image_directory = '../data/planet/france/april/'
label_directory = '../data/planet/france/labels/'

# Load train/val/test splits
splits_df = pd.read_csv('../data/splits/hanAndBurak_planetImagery_splits.csv')
train_names = splits_df[splits_df['fold'] == 'train']['image_id'].values
val_names = splits_df[splits_df['fold'] == 'val']['image_id'].values
test_names = splits_df[splits_df['fold'] == 'test']['image_id'].values

train_dataset = PlanetDataset(image_directory, label_directory, image_names=train_names)
val_dataset = PlanetDataset(image_directory, label_directory, image_names=val_names)
test_dataset = PlanetDataset(image_directory, label_directory, image_names=test_names)

train_dataloader = gluon.data.DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = gluon.data.DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = gluon.data.DataLoader(test_dataset, batch_size=batch_size)

# define model
model = ResUNet_d6(_nfilters_init=n_filters, _NClasses=n_classes)
model.initialize()
model.hybridize()
model.collect_params().reset_ctx(mx.gpu())

# define loss function
tanimoto_dual = Tanimoto_wth_dual()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(model.collect_params(),
                        'adam', {'learning_rate': lr})

# containers for metrics to log
train_metrics = {'train_loss': [], 'train_acc': [], 'train_f1': [], 
                 'train_mcc': []}
val_metrics = {'val_loss': [], 'val_acc': [], 'val_f1': [], 
               'val_mcc': []}
best_mcc = 0.0

# training loop
for epoch in range(1, epochs+1):
    
    # training set
    train_loss, train_accuracy, train_f1, train_mcc = train_model(
        train_dataloader, model, tanimoto_dual, trainer, epoch, args)
        
    # training set metrics
    train_loss_avg = train_loss / len(train_dataset)
    train_metrics['train_loss'].append(train_loss_avg)
    train_metrics['train_acc'].append(train_accuracy.get()[1])
    train_metrics['train_f1'].append(train_f1.get()[1])
    train_metrics['train_mcc'].append(train_mcc.get()[1])
    
    # validation set
    val_loss, val_accuracy, val_f1, val_mcc = evaluate_model(
        val_dataloader, model, tanimoto_dual, epoch, args)
    
    # validation set metrics
    val_loss_avg = val_loss / len(val_dataset)
    val_metrics['val_loss'].append(val_loss_avg)
    val_metrics['val_acc'].append(val_accuracy.get()[1])
    val_metrics['val_f1'].append(val_f1.get()[1])
    val_metrics['val_mcc'].append(val_mcc.get()[1])
    
    print("Epoch {}:".format(epoch))
    print("    Train loss {:0.3f}, accuracy {:0.3f}, F1-score {:0.3f}, MCC: {:0.3f}".format(
        train_loss_avg, train_accuracy.get()[1], train_f1.get()[1], train_mcc.get()[1]))
    print("    Val loss {:0.3f}, accuracy {:0.3f}, F1-score {:0.3f}, MCC: {:0.3f}".format(
        val_loss_avg, val_accuracy.get()[1], val_f1.get()[1], val_mcc.get()[1]))
    
    # save model based on best MCC metric
    if val_mcc.get()[1] > best_mcc:
        model.save_parameters(save_file_name)
        best_mcc = val_mcc.get()[1]

    # visdom
    vis.line(Y=np.stack([train_metrics['train_loss'], val_metrics['val_loss']], axis=1), 
             X=np.arange(1, epoch+1), win="Loss", 
             opts=dict(legend=['train loss', 'val loss'], markers=False, title="Losses",
                       xlabel="Epoch", ylabel="Loss")
            )

