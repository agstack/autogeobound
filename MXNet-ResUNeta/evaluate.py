import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon
from mxnet import autograd


def evaluate_model(val_dataloader, model, tanimoto_dual, epoch, args):
    
    # initialize metrics
    cumulative_loss = 0
    accuracy = mx.metric.Accuracy()
    f1 = mx.metric.F1()
    mcc = mx.metric.MCC()
    
    # validation set
    for batch_i, (img, extent, boundary, distance, hsv) in enumerate(
        tqdm(val_dataloader, desc='Validation epoch {}'.format(epoch))):

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
        
    return cumulative_loss, accuracy, f1, mcc
