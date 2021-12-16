from PyQt5 import QtWidgets
from ui.mainWindow import UI_MainWindow

import sys
from model import Model


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = UI_MainWindow()
        self.ui.setupUi(self)

        self.model = Model()

    #     self.ui.findCornersBtn.clicked.connect(lambda: self.find_corners_button_clicked())
    #
    # def find_corners_button_clicked(self):
    #     self.calibration_pm.click_find_corners()


app = QtWidgets.QApplication([])
window = MainWindow()
window.show()
sys.exit(app.exec_())
