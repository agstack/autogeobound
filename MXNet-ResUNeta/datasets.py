import numpy as np
import pandas as pd
import os
import imageio
import cv2

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


class PlanetDataset(gluon.data.Dataset):
    
    def __init__(self, image_directory, label_directory, image_names=None,
                 image_suffix='.jpeg', label_suffix='.png'):
        self.image_directory = image_directory
        self.label_directory = label_directory
        
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        
        if image_names is None:
            image_names = os.listdir(image_directory)
            self.image_names = [x.split('.')[0] for x in image_names]
        else:
            self.image_names = image_names
        
    def __getitem__(self, item):
        image_path = os.path.join(self.image_directory, 
                                  str(self.image_names[item]) + self.image_suffix)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # TODO: change this
        image = image[:256, :256]
        
        extent_path = os.path.join(self.label_directory, 
                                   str(self.image_names[item]) + self.label_suffix)
        extent_mask = imageio.imread(extent_path)
        # TODO: change this
        extent_mask = extent_mask[:256, :256] / 255
        
        # brightness augmentation
        image = np.minimum(
            np.random.uniform(low=0.8, high=1.25) * image, 255).astype(np.uint8)
        
        # rotation augmentation
        k = np.random.randint(low=0, high=4)
        image = np.rot90(image, k, axes=(0,1))
        extent_mask = np.rot90(extent_mask, k, axes=(0,1))
        
        # flip augmentation
        if np.random.uniform() > 0.5:
            image = np.flip(image, axis=0)
            extent_mask = np.flip(extent_mask, axis=0)
        if np.random.uniform() > 0.5:
            image = np.flip(image, axis=1)
            extent_mask = np.flip(extent_mask, axis=1)
        
        boundary_mask = get_boundary(extent_mask)
        distance_mask = get_distance(extent_mask)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        image = mx.nd.array(np.moveaxis(image, -1, 0))
        image_hsv = mx.nd.array(np.moveaxis(image_hsv, -1, 0)) / 255.
        
        extent_mask = mx.nd.array(np.expand_dims(extent_mask, 0))
        boundary_mask = mx.nd.array(np.expand_dims(boundary_mask, 0))
        distance_mask = mx.nd.array(np.expand_dims(distance_mask, 0))
        
        return image, extent_mask, boundary_mask, distance_mask, image_hsv
    
    def __len__(self):
        return len(self.image_names)

    
class PlanetMultitemp(gluon.data.Dataset):
    
    def __init__(self, image_directories, label_directory, image_names=None,
                 image_suffix='.jpeg', label_suffix='.png', n_channels=3):
        self.image_directories = image_directories
        self.label_directory = label_directory
        
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.n_channels = n_channels
        
        if image_names is None:
            image_names = os.listdir(image_directory[0])
            self.image_names = [x.split('.')[0] for x in image_names]
        else:
            self.image_names = image_names
        
    def __getitem__(self, item):
        
        images = np.zeros((256, 256, self.n_channels*len(self.image_directories)), dtype=np.float32)
        for i, image_directory in enumerate(self.image_directories):
            image_path = os.path.join(image_directory, 
                                      str(self.image_names[item]) + self.image_suffix)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            # TODO: change this
            image = image[:256, :256]
            
            # brightness augmentation
            image = np.minimum(
                np.random.uniform(low=0.8, high=1.25) * image, 255).astype(np.uint8)
            
            images[:,:,i*self.n_channels:(i+1)*self.n_channels] = image
        
        extent_path = os.path.join(self.label_directory, 
                                   str(self.image_names[item]) + self.label_suffix)
        extent_mask = imageio.imread(extent_path)
        # TODO: change this
        extent_mask = extent_mask[:256, :256] / 255
        
        # rotation augmentation
        k = np.random.randint(low=0, high=4)
        images = np.rot90(images, k, axes=(0,1))
        extent_mask = np.rot90(extent_mask, k, axes=(0,1))
        
        # flip augmentation
        if np.random.uniform() > 0.5:
            images = np.flip(images, axis=0)
            extent_mask = np.flip(extent_mask, axis=0)
        if np.random.uniform() > 0.5:
            images = np.flip(images, axis=1)
            extent_mask = np.flip(extent_mask, axis=1)
        
        boundary_mask = get_boundary(extent_mask)
        distance_mask = get_distance(extent_mask)
        
        images_hsv = np.zeros_like(images)
        for i in range(len(self.image_directories)):
            image = images[:,:,i*self.n_channels:(i+1)*self.n_channels]
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            images_hsv[:,:,i*self.n_channels:(i+1)*self.n_channels] = image_hsv
        
        images = mx.nd.array(np.moveaxis(images, -1, 0))
        images_hsv = mx.nd.array(np.moveaxis(images_hsv, -1, 0)) / 255.
        
        extent_mask = mx.nd.array(np.expand_dims(extent_mask, 0))
        boundary_mask = mx.nd.array(np.expand_dims(boundary_mask, 0))
        distance_mask = mx.nd.array(np.expand_dims(distance_mask, 0))
        
        return images, extent_mask, boundary_mask, distance_mask, images_hsv
    
    def __len__(self):
        return len(self.image_names)
    

class PlanetDatasetWithClasses(gluon.data.Dataset):
    
    def __init__(self, image_directory, label_directory, fold='train', image_names=None, 
                 label_names=None, classes=[255], image_suffix='.jpeg', label_suffix='.png',
                 boundary_kernel_size=(3,3)):
        self.image_directory = image_directory
        self.label_directory = label_directory
        
        self.fold = fold
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.classes = classes
        
        if image_names is None:
            image_names = os.listdir(image_directory)
            self.image_names = [x.split('.')[0] for x in image_names]
        else:
            self.image_names = image_names
            
        if label_names is None:
            self.label_names = image_names
        else:
            self.label_names = label_names
            
        self.boundary_kernel_size = boundary_kernel_size
        
    def __getitem__(self, item):
        image = np.zeros((256,256,3))
        image_path = os.path.join(self.image_directory, 
                                  str(self.image_names[item]) + self.image_suffix)
        image_temp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        nrow, ncol, nchannels = image_temp.shape
        # TODO: change this
        image[:nrow, :ncol] = image_temp[:256, :256]
        
        extent_path = os.path.join(self.label_directory, 
                                   str(self.label_names[item % len(self.label_names)]) + self.label_suffix)
        extent_mask = np.zeros((256,256))
        extent_image = imageio.imread(extent_path)
        nrow, ncol = extent_image.shape
        # TODO: change this
        extent_image = np.array(np.isin(extent_image, self.classes), dtype=np.uint8)
        extent_mask[:nrow, :ncol] = extent_image[:256, :256]
#         extent_mask = extent_mask[:256, :256] / 255
        
        if self.fold == 'train':
            # brightness augmentation
            image = np.minimum(
                np.random.uniform(low=0.8, high=1.25) * image, 255)

            # rotation augmentation
            k = np.random.randint(low=0, high=4)
            image = np.rot90(image, k, axes=(0,1))
            extent_mask = np.rot90(extent_mask, k, axes=(0,1))

            # flip augmentation
            if np.random.uniform() > 0.5:
                image = np.flip(image, axis=0)
                extent_mask = np.flip(extent_mask, axis=0)
            if np.random.uniform() > 0.5:
                image = np.flip(image, axis=1)
                extent_mask = np.flip(extent_mask, axis=1)
        
        image = image.astype(np.uint8)
        boundary_mask = get_boundary(extent_mask, kernel_size=self.boundary_kernel_size)
        distance_mask = get_distance(extent_mask)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        image = mx.nd.array(np.moveaxis(image, -1, 0))
        image_hsv = mx.nd.array(np.moveaxis(image_hsv, -1, 0)) / 255.
        
        extent_mask = mx.nd.array(np.expand_dims(extent_mask, 0))
        boundary_mask = mx.nd.array(np.expand_dims(boundary_mask, 0))
        distance_mask = mx.nd.array(np.expand_dims(distance_mask, 0))
        
        return image, extent_mask, boundary_mask, distance_mask, image_hsv
    
    def __len__(self):
        return len(self.image_names)
    
    
class PlanetDatasetWithClassesFullPaths(gluon.data.Dataset):
    
    def __init__(self, fold='train', image_names=None, label_names=None, classes=[1],
                 noncrop_classes=[2], boundary_kernel_size=(3,3)):
        
        self.fold = fold
        self.classes = classes
        self.noncrop_classes = noncrop_classes
        self.image_names = image_names
        self.label_names = label_names
        self.boundary_kernel_size = boundary_kernel_size
        
    def __getitem__(self, item):
        image = np.zeros((256,256,3))
        image_path = self.image_names[item]
        image_temp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        nrow, ncol, nchannels = image_temp.shape
        
        extent_path = self.label_names[item]
        extent_mask = np.zeros((256,256))
        extent_original = imageio.imread(extent_path)
        nrow, ncol = extent_original.shape
        extent_image = np.array(np.isin(extent_original, self.classes), dtype=np.uint8)
        
        # TODO: change this
        # extent_mask[:nrow, :ncol] = extent_image[:256, :256]

        topleft_x, topleft_y = 0, 0
        if nrow > 256:
            topleft_x = np.random.choice(np.arange(0, nrow-256))
        if ncol > 256:
            topleft_y = np.random.choice(np.arange(0, ncol-256))

        image[:nrow, :ncol] = image_temp[topleft_x:topleft_x+256, topleft_y:topleft_y+256]
        extent_mask[:nrow, :ncol] = extent_image[topleft_x:topleft_x+256, topleft_y:topleft_y+256]
        
        if self.fold == 'train':
            # brightness augmentation
            image = np.minimum(
                np.random.uniform(low=0.8, high=1.25) * image, 255)

            # rotation augmentation
            k = np.random.randint(low=0, high=4)
            image = np.rot90(image, k, axes=(0,1))
            extent_mask = np.rot90(extent_mask, k, axes=(0,1))

            # flip augmentation
            if np.random.uniform() > 0.5:
                image = np.flip(image, axis=0)
                extent_mask = np.flip(extent_mask, axis=0)
            if np.random.uniform() > 0.5:
                image = np.flip(image, axis=1)
                extent_mask = np.flip(extent_mask, axis=1)
        
        image = image.astype(np.uint8)
        boundary_mask = get_boundary(extent_mask, kernel_size=self.boundary_kernel_size)
        distance_mask = get_distance(extent_mask)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        image = mx.nd.array(np.moveaxis(image, -1, 0))
        image_hsv = mx.nd.array(np.moveaxis(image_hsv, -1, 0)) / 255.
        
        extent_mask = mx.nd.array(np.expand_dims(extent_mask, 0))
        boundary_mask = mx.nd.array(np.expand_dims(boundary_mask, 0))
        distance_mask = mx.nd.array(np.expand_dims(distance_mask, 0))
        
        return image, extent_mask, boundary_mask, distance_mask, image_hsv
    
    def __len__(self):
        return len(self.image_names)
    
    
class PlanetDatasetWithClassesFullPathsMasked(gluon.data.Dataset):
    
    def __init__(self, fold='train', image_names=None, label_names=None, classes=[1],
                 noncrop_classes=[2], boundary_kernel_size=(3,3)):
        
        self.fold = fold
        self.classes = classes
        self.noncrop_classes = noncrop_classes
        self.image_names = image_names
        self.label_names = label_names
        self.boundary_kernel_size = boundary_kernel_size
        
    def __getitem__(self, item):
        image = np.zeros((256,256,3))
        image_path = self.image_names[item]
        image_temp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        nrow, ncol, nchannels = image_temp.shape
        
        extent_path = self.label_names[item]
        extent_mask = np.zeros((256,256))
        extent_original = imageio.imread(extent_path)
        nrow, ncol = extent_original.shape
        extent_image = np.array(np.isin(extent_original, self.classes), dtype=np.uint8)
        
        # TODO: change this
        # extent_mask[:nrow, :ncol] = extent_image[:256, :256]

        topleft_x, topleft_y = 0, 0
        if nrow > 256:
            topleft_x = np.random.choice(np.arange(0, nrow-256))
        if ncol > 256:
            topleft_y = np.random.choice(np.arange(0, ncol-256))

        image[:nrow, :ncol] = image_temp[topleft_x:topleft_x+256, topleft_y:topleft_y+256]
        extent_mask[:nrow, :ncol] = extent_image[topleft_x:topleft_x+256, topleft_y:topleft_y+256]
        
        noncrop_mask = np.zeros((256,256))
        noncrop_image = np.array(np.isin(extent_original, self.noncrop_classes), dtype=np.uint8)
        # noncrop_mask[:nrow, :ncol] = noncrop_image[:256, :256]
        noncrop_mask[:nrow, :ncol] = noncrop_image[topleft_x:topleft_x+256, topleft_y:topleft_y+256]
        
        if self.fold == 'train':
            # brightness augmentation
            image = np.minimum(
                np.random.uniform(low=0.8, high=1.25) * image, 255)

            # rotation augmentation
            k = np.random.randint(low=0, high=4)
            image = np.rot90(image, k, axes=(0,1))
            extent_mask = np.rot90(extent_mask, k, axes=(0,1))

            # flip augmentation
            if np.random.uniform() > 0.5:
                image = np.flip(image, axis=0)
                extent_mask = np.flip(extent_mask, axis=0)
            if np.random.uniform() > 0.5:
                image = np.flip(image, axis=1)
                extent_mask = np.flip(extent_mask, axis=1)
        
        image = image.astype(np.uint8)
        boundary_mask = get_boundary(extent_mask, kernel_size=self.boundary_kernel_size)
        distance_mask = get_distance(extent_mask)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # define label mask
        label_mask = np.array((extent_mask + boundary_mask + noncrop_mask) >= 1, dtype=np.float32)
        
        image = mx.nd.array(np.moveaxis(image, -1, 0))
        image_hsv = mx.nd.array(np.moveaxis(image_hsv, -1, 0)) / 255.
        
        extent_mask = mx.nd.array(np.expand_dims(extent_mask, 0))
        boundary_mask = mx.nd.array(np.expand_dims(boundary_mask, 0))
        distance_mask = mx.nd.array(np.expand_dims(distance_mask, 0))
        label_mask = mx.nd.array(np.expand_dims(label_mask, 0))
        
        return image, extent_mask, boundary_mask, distance_mask, image_hsv, label_mask
    
    def __len__(self):
        return len(self.image_names)
    
    
class PlanetMultitempWithClasses(gluon.data.Dataset):
    
    def __init__(self, image_directories, label_directory, fold='train', image_names=None, classes=[255],
                 image_suffixes=['.jpeg', '.jpeg'], label_suffix='.png', n_channels=3):
        self.image_directories = image_directories
        self.label_directory = label_directory
        
        self.fold = fold
        self.image_suffixes = image_suffixes
        self.label_suffix = label_suffix
        self.n_channels = n_channels
        self.classes = classes
        
        if image_names is None:
            image_names = os.listdir(image_directory[0])
            self.image_names = [x.split('.')[0] for x in image_names]
        else:
            self.image_names = image_names
        
    def __getitem__(self, item):
        
        images = np.zeros((256, 256, self.n_channels*len(self.image_directories)), dtype=np.float32)
        for i, image_directory in enumerate(self.image_directories):
            image = np.zeros((256, 256, self.n_channels))
            image_path = os.path.join(image_directory, 
                                      str(self.image_names[item]) + self.image_suffixes[i])
            image_temp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            nrow, ncol, nchannels = image_temp.shape
            # TODO: change this
            image[:nrow, :ncol] = image_temp[:256, :256]
            
            if self.fold == 'train':
                # brightness augmentation
                image = np.minimum(
                    np.random.uniform(low=0.8, high=1.25) * image, 255)
            
            images[:,:,i*self.n_channels:(i+1)*self.n_channels] = image
        
        extent_mask = np.zeros((256,256))
        extent_path = os.path.join(self.label_directory, 
                                   str(self.image_names[item]) + self.label_suffix)
        extent_image = imageio.imread(extent_path)
        # TODO: change this
        extent_image = np.array(np.isin(extent_image, self.classes), dtype=np.uint8)
        extent_mask[:nrow, :ncol] = extent_image[:256, :256]
        
        if self.fold == 'train':
            # rotation augmentation
            k = np.random.randint(low=0, high=4)
            images = np.rot90(images, k, axes=(0,1))
            extent_mask = np.rot90(extent_mask, k, axes=(0,1))

            # flip augmentation
            if np.random.uniform() > 0.5:
                images = np.flip(images, axis=0)
                extent_mask = np.flip(extent_mask, axis=0)
            if np.random.uniform() > 0.5:
                images = np.flip(images, axis=1)
                extent_mask = np.flip(extent_mask, axis=1)
        
        images = images.astype(np.uint8)
        boundary_mask = get_boundary(extent_mask)
        distance_mask = get_distance(extent_mask)
        
        images_hsv = np.zeros_like(images)
        for i in range(len(self.image_directories)):
            image = images[:,:,i*self.n_channels:(i+1)*self.n_channels]
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            images_hsv[:,:,i*self.n_channels:(i+1)*self.n_channels] = image_hsv
        
        images = mx.nd.array(np.moveaxis(images, -1, 0))
        images_hsv = mx.nd.array(np.moveaxis(images_hsv, -1, 0)) / 255.
        
        extent_mask = mx.nd.array(np.expand_dims(extent_mask, 0))
        boundary_mask = mx.nd.array(np.expand_dims(boundary_mask, 0))
        distance_mask = mx.nd.array(np.expand_dims(distance_mask, 0))
        
        return images, extent_mask, boundary_mask, distance_mask, images_hsv
    
    def __len__(self):
        return len(self.image_names)
    
    
    
class PlanetDatasetNoLabels(gluon.data.Dataset):
    
    def __init__(self, image_directory, image_names=None, image_suffix='.jpeg'):
        self.image_directory = image_directory
        self.image_suffix = image_suffix
        
        if image_names is None:
            image_names = os.listdir(image_directory)
            self.image_names = [x.split('.')[0] for x in image_names]
        else:
            self.image_names = image_names
        
    def __getitem__(self, item):
        image = np.zeros((256,256,3))
        image_path = os.path.join(self.image_directory, 
                                  str(self.image_names[item]) + self.image_suffix)
        #print(image_path)
        image_temp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        nrow, ncol, nchannels = image_temp.shape
        # TODO: change this
        image[:nrow, :ncol] = image_temp[:256, :256]
        image = image.astype(np.uint8)
        image = mx.nd.array(np.moveaxis(image, -1, 0))
        
        return image
    
    def __len__(self):
        return len(self.image_names)
    

class AirbusMasked(gluon.data.Dataset):
    
    def __init__(self, fold='train', image_names=None, label_names=None, classes=[1],
                 noncrop_classes=[2], boundary_kernel_size=(3,3), random_crop=True):
        
        self.fold = fold
        self.classes = classes
        self.noncrop_classes = noncrop_classes
        self.image_names = image_names
        self.label_names = label_names
        self.boundary_kernel_size = boundary_kernel_size
        self.random_crop = random_crop
        
    def __getitem__(self, item):
        
        # take random crop of Airbus image
        got_crop = False
        attempts = 0
        image = np.zeros((256,256,3))
        image_path = self.image_names[item]
        image_temp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        nrow, ncol, nchannels = image_temp.shape
        
        while not got_crop:
            topleft_x = 0
            topleft_y = 0
            if nrow > 256 and self.random_crop:
                topleft_x = np.random.choice(np.arange(0, nrow-256))
            if ncol > 256 and self.random_crop:
                topleft_y = np.random.choice(np.arange(0, ncol-256))
            image[:nrow, :ncol] = image_temp[topleft_x:topleft_x+256, topleft_y:topleft_y+256]

            extent_path = self.label_names[item]
            extent_mask = np.zeros((256,256))
            extent_original = imageio.imread(extent_path)
            nrow, ncol = extent_original.shape
        
            extent_image = np.array(np.isin(extent_original, self.classes), dtype=np.uint8)
            extent_mask[:nrow, :ncol] = extent_image[topleft_x:topleft_x+256, topleft_y:topleft_y+256]
            if np.sum(extent_mask) > 0 or attempts >= 100:
                got_crop = True
            attempts += 1
        
            noncrop_mask = np.zeros((256,256))
            noncrop_image = np.array(np.isin(extent_original, self.noncrop_classes), dtype=np.uint8)
            noncrop_mask[:nrow, :ncol] = noncrop_image[topleft_x:topleft_x+256, topleft_y:topleft_y+256]
        
        if self.fold == 'train':
            # brightness augmentation
            image = np.minimum(
                np.random.uniform(low=0.8, high=1.25) * image, 255)

            # rotation augmentation
            k = np.random.randint(low=0, high=4)
            image = np.rot90(image, k, axes=(0,1))
            extent_mask = np.rot90(extent_mask, k, axes=(0,1))

            # flip augmentation
            if np.random.uniform() > 0.5:
                image = np.flip(image, axis=0)
                extent_mask = np.flip(extent_mask, axis=0)
            if np.random.uniform() > 0.5:
                image = np.flip(image, axis=1)
                extent_mask = np.flip(extent_mask, axis=1)
        
        image = image.astype(np.uint8)
        boundary_mask = get_boundary(extent_mask, kernel_size=self.boundary_kernel_size)
        distance_mask = get_distance(extent_mask)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # define label mask
        label_mask = np.array((extent_mask + boundary_mask + noncrop_mask) >= 1, dtype=np.float32)
        
        image = mx.nd.array(np.moveaxis(image, -1, 0))
        image_hsv = mx.nd.array(np.moveaxis(image_hsv, -1, 0)) / 255.
        
        extent_mask = mx.nd.array(np.expand_dims(extent_mask, 0))
        boundary_mask = mx.nd.array(np.expand_dims(boundary_mask, 0))
        distance_mask = mx.nd.array(np.expand_dims(distance_mask, 0))
        label_mask = mx.nd.array(np.expand_dims(label_mask, 0))
        
        return image, extent_mask, boundary_mask, distance_mask, image_hsv, label_mask
    
    def __len__(self):
        return len(self.image_names)
    
    
class AirbusNoLabels(gluon.data.Dataset):
    
    def __init__(self, image_directory, image_names=None, image_suffix='.jpeg'):
        self.image_directory = image_directory
        self.image_suffix = image_suffix
        
        if image_names is None:
            image_names = os.listdir(image_directory)
            self.image_names = [x.split('.')[0] for x in image_names]
        else:
            self.image_names = image_names
        
    def __getitem__(self, item):
        image = np.zeros((256,256,3))
        image_path = os.path.join(self.image_directory, 
                                  str(self.image_names[item]) + self.image_suffix)
        #print(image_path)
        image_temp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        nrow, ncol, nchannels = image_temp.shape
        # TODO: change this
        image[:nrow, :ncol] = image_temp[(nrow-256)//2:(nrow-256)//2+256, (ncol-256)//2:(ncol-256)//2+256]
        image = image.astype(np.uint8)
        image = mx.nd.array(np.moveaxis(image, -1, 0))
        
        return image
    
    def __len__(self):
        return len(self.image_names)
    

class MultitempMasked(gluon.data.Dataset):
    
    def __init__(self, image_directories, fold='train', image_names=None, label_names=None, classes=[1],
                 noncrop_classes=[2], boundary_kernel_size=(3,3), n_channels=3, image_suffixes=None,
                 shuffle_directories=False):
        
        self.fold = fold
        self.classes = classes
        self.noncrop_classes = noncrop_classes
        self.image_names = image_names
        self.label_names = label_names
        self.boundary_kernel_size = boundary_kernel_size
        self.n_channels = n_channels
        self.image_directories = image_directories
        self.image_suffixes = image_suffixes
        self.shuffle_directories = shuffle_directories
        
    def __getitem__(self, item):
        
        # take random crop of image
        got_crop = False
        tries = 0
        while not got_crop:
            
            extent_path = self.label_names[item]
            extent_mask = np.zeros((256,256))
            extent_original = imageio.imread(extent_path)
            
            topleft_x, topleft_y = 0, 0
            nrow, ncol = extent_original.shape
            if nrow > 256:
                topleft_x = np.random.choice(np.arange(0, nrow-256))
            if ncol > 256:
                topleft_y = np.random.choice(np.arange(0, ncol-256))
        
            extent_image = np.array(np.isin(extent_original, self.classes), dtype=np.uint8)
            extent_mask[:nrow, :ncol] = extent_image[topleft_x:topleft_x+256, topleft_y:topleft_y+256]
            if np.sum(extent_mask) > 0 or tries > 100:
                got_crop = True
            else: # go back and sample a x and y again
                tries += 1
                continue
            
            images = np.zeros((256, 256, self.n_channels * len(self.image_directories)), dtype=np.float32)
            images_hsv = np.zeros_like(images)
            
            indices = list(np.arange(len(self.image_directories)))
            img_dirs = np.copy(self.image_directories)
            image_suffixes = np.copy(self.image_suffixes)
            if self.shuffle_directories:
                np.random.shuffle(indices)
                img_dirs = img_dirs[indices]
                image_suffixes = image_suffixes[indices]
                
            for i, image_directory in enumerate(img_dirs):
                image = np.zeros((256, 256, self.n_channels))
                image_hsv = np.zeros_like(image)
                image_path = os.path.join(image_directory, self.image_names[item] + image_suffixes[i])
                image_temp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                nrow, ncol, nchannels = image_temp.shape
   
                image[:nrow, :ncol] = image_temp[topleft_x:topleft_x+256, topleft_y:topleft_y+256]
                image_hsv[:nrow, :ncol] = cv2.cvtColor(np.array(image[:nrow, :ncol], dtype=np.uint8), cv2.COLOR_RGB2HSV)
                
                if self.fold == 'train':
                    # brightness augmentation
                    image = np.minimum(
                        np.random.uniform(low=0.8, high=1.25) * image, 255)
                
                images[:,:,i*self.n_channels:(i+1)*self.n_channels] = image
                images_hsv[:,:,i*self.n_channels:(i+1)*self.n_channels] = image_hsv
        
            noncrop_mask = np.zeros((256,256))
            noncrop_image = np.array(np.isin(extent_original, self.noncrop_classes), dtype=np.uint8)
            noncrop_mask[:nrow, :ncol] = noncrop_image[topleft_x:topleft_x+256, topleft_y:topleft_y+256]
        
        if self.fold == 'train':

            # rotation augmentation
            k = np.random.randint(low=0, high=4)
            images = np.rot90(images, k, axes=(0,1))
            extent_mask = np.rot90(extent_mask, k, axes=(0,1))

            # flip augmentation
            if np.random.uniform() > 0.5:
                images = np.flip(images, axis=0)
                extent_mask = np.flip(extent_mask, axis=0)
            if np.random.uniform() > 0.5:
                images = np.flip(images, axis=1)
                extent_mask = np.flip(extent_mask, axis=1)
        
        images = images.astype(np.uint8)
        boundary_mask = get_boundary(extent_mask, kernel_size=self.boundary_kernel_size)
        distance_mask = get_distance(extent_mask)
        
        # define label mask
        label_mask = np.array((extent_mask + boundary_mask + noncrop_mask) >= 1, dtype=np.float32)
        
        images = mx.nd.array(np.moveaxis(images, -1, 0))
        images_hsv = mx.nd.array(np.moveaxis(images_hsv, -1, 0)) / 255.
        
        extent_mask = mx.nd.array(np.expand_dims(extent_mask, 0))
        boundary_mask = mx.nd.array(np.expand_dims(boundary_mask, 0))
        distance_mask = mx.nd.array(np.expand_dims(distance_mask, 0))
        label_mask = mx.nd.array(np.expand_dims(label_mask, 0))
        
        return images, extent_mask, boundary_mask, distance_mask, images_hsv, label_mask
    
    def __len__(self):
        return len(self.image_names)
