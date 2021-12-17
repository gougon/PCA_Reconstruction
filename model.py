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
        return np.array(imgs)

    def __flatten_imgs(self, imgs):
        flt_imgs = []
        for img in imgs:
            flt_imgs.append(img.flatten())
        return np.array(flt_imgs)

    def __reverse_imgs(self, flt_imgs):
        imgs = []
        for img in flt_imgs:
            imgs.append(img.reshape(400, 400, 3))
        return np.array(imgs)

    def reconstruction(self):
        img_pca = self.__pca.fit_transform(self.__flt_imgs)
        recons = self.__pca.inverse_transform(img_pca)
        return self.__reverse_imgs(recons)

    def get_images(self):
        return self.__images

    def compute_reconstruction_error(self):
        recons = self.reconstruction()
        if np.max(recons) > 1:
            recons *= 255
        imgs = self.__images
        if np.max(imgs) > 1:
            imgs *= 255

        g_recons, g_imgs = [], []
        for img, recon in zip(imgs, recons):
            g_recons.append(cv2.cvtColor(recon.astype('uint8'), cv2.COLOR_RGB2GRAY))
            g_imgs.append(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY))
        g_recons = np.array(g_recons)
        g_imgs = np.array(g_imgs)

        re = [0] * 30
        for i, (img, recon) in enumerate(zip(g_imgs, g_recons)):
            for i_row, r_row in zip(img, recon):
                for i_pixel, r_pixel in zip(i_row, r_row):
                    re[i] += abs(i_pixel - r_pixel)
        re = np.array(re)
        return re
