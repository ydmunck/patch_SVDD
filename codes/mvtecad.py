import numpy as np
from PIL import Image
from imageio import imread
from glob import glob
from sklearn.metrics import roc_auc_score
import os
from rvai.types import String, Image

DATASET_PATH = "/workspace/rvai_algorithms/cells/anomaly/patch_svdd/rvai/cells/patch_svdd/data/"

__all__ = ['objs',
           'get_x', 'get_x_standardized']


def gray2rgb(images):
    tile_shape = tuple(np.ones(len(images.shape), dtype=int))
    tile_shape += (3,)

    images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
    # print(images.shape)
    return images


class LoadImages:
    def __init__(self, obj: String):
        """Initializes the data
        :param obj: type object used for anomaly detection
        :type obj: String
        """
        self.obj = obj
        self.images_train = None
        self.images_test = None
        self.standardized_images_train = None
        self.standardized_images_test = None
        self.mean = None
    
    def get_standardized_images_train(self) -> np.asarray:
        """Get standardized images for the train dataset
        :return: Standardized value for the images of the train dataset
        :rtype: np.asarray
        """
        if self.standardized_images_train is None:
            self.standardized_images_train = (self.get_images_train().astype(np.float32) - self.get_mean()) / 255
        return self.standardized_images_train
    
    def get_standardized_images_test(self) -> np.asarray:
        """Get standardized images for the test dataset
        :param image: input image for inference
        :type image: Image
        :return: Standardized value for the images of the test dataset
        :rtype: np.asarray
        """
        if self.standardized_images_test is None:
            self.standardized_images_test = (self.get_images_test().astype(np.float32) - self.get_mean()) / 255
        return self.standardized_images_test
    
    def get_mean(self):
        """Get mean value of images of the train dataset
        :return: mean value
        :rtype: np.asarray
        """
        if self.mean is None:
            self.mean = self.get_images_train().astype(np.float32).mean(axis=0)
        return self.mean
    
    def get_images_train(self) -> np.asarray:
        """Load images in memory for the train dataset
        :param obj: type object used for anomaly detection
        :type obj: String
        :return: images of the train dataset
        :rtype: np.asarray
        """
        if self.images_train is None:
            mode = "train"
            fpattern = os.path.join(DATASET_PATH, f'{self.obj}/{mode}/*/*.jpg')
            fpaths = sorted(glob(fpattern))
    
            images = np.asarray(list(map(imread, fpaths)))

            IF_GRAYSCALE = images.shape[-1] != 3
            if IF_GRAYSCALE:
                images = gray2rgb(images)
                
            self.images_train = np.asarray(images)
        return self.images_train
    
    def get_images_test(self) -> np.asarray:
        """Load image in memory for testing
        :param image: input image for inference
        :type image: Image
        :return: image for testing
        :rtype: np.asarray
        """
        if self.images_test is None:
            mode = "test"
            fpattern = os.path.join(DATASET_PATH, f'{self.obj}/{mode}/*/*.jpg')
            fpaths = sorted(glob(fpattern))
            images = np.asarray(list(map(imread, fpaths)))
            IF_GRAYSCALE = images.shape[-1] != 3
            if IF_GRAYSCALE:
                images = gray2rgb(images)
                
            self.images_test = np.asarray(images)
        return self.images_test
