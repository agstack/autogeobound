import numpy as np
import pandas as pd
import cv2
import visdom

def visdom_visualize_batch(img, extent, boundary, distance,
                           extent_pred, boundary_pred, distance_pred,
                           hsv_pred, title="Train images"):

    img, extent, boundary, distance = img.asnumpy(), extent.asnumpy(), boundary.asnumpy(), distance.asnumpy()
    extent_pred, boundary_pred = extent_pred.asnumpy(), boundary_pred.asnumpy()
    hsv_pred = hsv_pred.asnumpy()

    # put everything in one window
    batch_size, nchannels, nrows, ncols = img.shape
    padding = 10
    result = np.zeros((3, 8*nrows + 7*padding, batch_size*ncols + (batch_size-1)*padding))

    for j, item in enumerate([img, hsv_pred, extent, extent_pred, boundary, boundary_pred,
                              distance, distance_pred]):
        if item.shape[1] == 1:
            item = np.tile(item, (1,3,1,1))
        if j == 1: # convert HSV to RGB
            item = np.moveaxis(item, 0, -1)
            item = cv2.cvtColor(item, cv2.COLOR_HSV2RGB)
            item = np.moveaxis(item, -1, 0)
        for i in range(batch_size):
            result[:, j*(nrows+padding):(j+1)*nrows+j*padding, i*(ncols+padding):(i+1)*ncols+i*padding] = item[i]

    args['visdom'].images(result, nrow=4, win=title, opts={'title': title})
