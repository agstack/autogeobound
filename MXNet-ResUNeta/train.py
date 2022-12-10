import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon
from mxnet import autograd

from visdom import *


def train_model(train_dataloader, model, tanimoto_dual, trainer, epoch, args):
    
    # initialize metrics
    cumulative_loss = 0
    accuracy = mx.metric.Accuracy()
    f1 = mx.metric.F1()
    mcc = mx.metric.MCC()
    
    # training set
    for batch_i, (img, extent, boundary, distance, hsv) in enumerate(
        tqdm(train_dataloader, desc='Training epoch {}'.format(epoch))):
        
        with autograd.record():

            img = img.as_in_context(mx.gpu())
            extent = extent.as_in_context(mx.gpu())
            boundary = boundary.as_in_context(mx.gpu())
            distance = distance.as_in_context(mx.gpu())
            hsv = hsv.as_in_context(mx.gpu())
            
            logits, bound, dist, convc = model(img)
            
            # multi-task loss
            # TODO: wrap this in a custom loss function / class
            loss_extent = mx.nd.sum(1 - tanimoto_dual(logits, extent))
            loss_boundary = mx.nd.sum(1 - tanimoto_dual(bound, boundary))
            loss_distance = mx.nd.sum(1 - tanimoto_dual(dist, distance))
            loss_hsv = mx.nd.sum(1 - tanimoto_dual(convc, hsv))
            
            loss = 0.25 * (loss_extent + loss_boundary + loss_distance + loss_hsv)
            
        loss.backward()
        trainer.step(args['batch_size'])
        
        # update metrics based on every batch
        cumulative_loss += mx.nd.sum(loss).asscalar()
        # accuracy
        extent_predicted_classes = mx.nd.ceil(logits[:,[0],:,:] - 0.5)
        accuracy.update(extent, extent_predicted_classes)
        # f1 score
        prediction = logits[:,0,:,:].reshape(-1)
        probabilities = mx.nd.stack(1 - prediction, prediction, axis=1)
        f1.update(extent.reshape(-1), probabilities)
        # MCC metric
        mcc.update(extent.reshape(-1), probabilities)
        # TODO: eccentricity
        # TODO: ...
        
        visdom_visualize_batch(img, extent, boundary, distance,
                               logits, bound, dist, convc)
        #if batch_i % args['visdom_every'] == 0:
        #    args['visdom'].images(img.asnumpy(), nrow=4, win="train_images", opts={'title': "Train images"})
        #    args['visdom'].images(extent.asnumpy(), nrow=4, win="train_extent", opts={'title': "Train extent (ground truth)"})
        #    args['visdom'].images(logits[:,[0]].asnumpy(), nrow=4, win="train_extpred", opts={'title': "Train extent (predicted)"})
        #    args['visdom'].images(distance.asnumpy(), nrow=4, win="train_distance", opts={'title': "Train distance (ground truth)"})
        #    args['visdom'].images(dist[:,[0]].asnumpy(), nrow=4, win="train_distpred", opts={'title': "Train distance (predicted)"})

    return cumulative_loss, accuracy, f1, mcc
