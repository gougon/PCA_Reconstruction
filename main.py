import numpy as np
from PyQt5 import QtWidgets
from ui.mainWindow import UI_MainWindow
import cv2
import matplotlib.pyplot as plt

import sys
from model import Model


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = UI_MainWindow()
        self.ui.setupUi(self)

        self.model = Model()

        self.ui.reconstructionButton.clicked.connect(lambda: self.reconstruction_button_clicked())
        self.ui.errorButton.clicked.connect(lambda: self.error_button_clicked())

    def reconstruction_button_clicked(self):
        imgs = self.model.get_images()
        recons = self.model.reconstruction()

        fig, axs = plt.subplots(4, 15, figsize=(15, 7))
        for y in range(4):
            for x in range(15):
                idx = int(int(y / 2) * 15 + x)
                axs[y, x].get_xaxis().set_visible(False)
                axs[y, x].get_yaxis().set_visible(False)
                if y % 2 == 0:
                    axs[y, x].imshow(imgs[idx] / 255)
                else:
                    axs[y, x].imshow(recons[idx] / 255)
        plt.show()

    def error_button_clicked(self):
        re = self.model.compute_reconstruction_error()
        print(re)


app = QtWidgets.QApplication([])
window = MainWindow()
window.show()
sys.exit(app.exec_())
