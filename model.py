import cv2
import numpy as np
import os
from sklearn.decomposition import PCA

import constant as const
import config as cfg


class Model():
    def __init__(self):
        self.__images = self.__load_images(const.DATASET_LOC)
        self.__flt_imgs = self.__flatten_imgs(self.__images)
        self.__pca = PCA(n_components=cfg.N_COMPONENTS)

    def __load_images(self, dir):
        imgs = []
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                imgs.append(cv2.imread(dirpath + filename)[..., ::-1])
        return imgs

    def __flatten_imgs(self, imgs):
        flt_imgs = []
        for img in imgs:
            flt_imgs.append(img.flatten())
        return np.array(flt_imgs)

    def __reverse_imgs(self, flt_imgs):
        imgs = []
        for img in flt_imgs:
            imgs.append(img.reshape(400, 400, 3))
        return imgs

    def reconstruction(self):
        img_pca = self.__pca.fit_transform(self.__flt_imgs)
        recons = self.__pca.inverse_transform(img_pca)
        return self.__reverse_imgs(recons)

    def get_images(self):
        return self.__images
