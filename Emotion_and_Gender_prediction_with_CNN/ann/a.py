# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'a.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from tensorflow import keras
import numpy as np
from keras.preprocessing import image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from keras.applications.imagenet_utils import preprocess_input

class Ui_MainWindow(object):
    def tahmin_bulun(self):
        model2 = keras.models.load_model('./cinsiyet_model.h5')
        image_path = dosyaYolum
        test_image = image.load_img(image_path, target_size=(250, 250), grayscale=False)
        test_data = image.img_to_array(test_image)
        test_data = np.expand_dims(test_data, axis=0)
        img_preprocessed = preprocess_input(test_data)
        prediction = model2.predict(img_preprocessed)
        if(prediction[0] == 1):
            self.lbl_tahmin_grafik.setText("fotodaki erkek")
        else:
            self.lbl_tahmin_grafik.setText("fotodaki kadin")
        model = keras.models.load_model('./facee_model.h5')
        image_path = dosyaYolum
        test_image = image.load_img(image_path, target_size=(48, 48), color_mode="grayscale")
        test_data = image.img_to_array(test_image)
        test_data = np.expand_dims(test_data, axis=0)
        test_data = np.vstack([test_data])
        results = model.predict(test_data, batch_size=1)
        class_names = ['kizgin', 'igrenme', 'korku', 'mutlu', 'uzgun', 'sasirma', 'dogal']
        a = np.argmax(results)
        self.lbl_tahmin_yazi.setText("sınıflandırma sonucu en yüksek oranla : "+ class_names[np.argmax(results)])
    def open_dialog_box(self):
        global dosyaYolum 
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        dosyaYolum = path
        self.lbl_photo.setPixmap(QtGui.QPixmap(dosyaYolum))
        print(dosyaYolum)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(975, 706)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnTahmin = QtWidgets.QPushButton(self.centralwidget)
        self.btnTahmin.setGeometry(QtCore.QRect(760, 200, 161, 71))
        self.btnTahmin.setObjectName("btnTahmin")
        self.btnTahmin.clicked.connect(self.tahmin_bulun)
        self.btnResimSec = QtWidgets.QPushButton(self.centralwidget)
        self.btnResimSec.setGeometry(QtCore.QRect(760, 110, 161, 71))
        self.btnResimSec.setObjectName("btnResimSec")
        self.btnResimSec.clicked.connect(self.open_dialog_box)
        self.lbl_photo = QtWidgets.QLabel(self.centralwidget)
        self.lbl_photo.setGeometry(QtCore.QRect(200, 40, 321, 281))
        self.lbl_photo.setScaledContents(True)
        self.lbl_photo.setObjectName("lbl_photo")
        self.lbl_tahmin_yazi = QtWidgets.QLabel(self.centralwidget)
        self.lbl_tahmin_yazi.setGeometry(QtCore.QRect(20, 430, 381, 191))
        self.lbl_tahmin_yazi.setObjectName("lbl_tahmin_yazi")
        self.lbl_tahmin_grafik = QtWidgets.QLabel(self.centralwidget)
        self.lbl_tahmin_grafik.setGeometry(QtCore.QRect(530, 425, 411, 191))
        self.lbl_tahmin_grafik.setObjectName("lbl_tahmin_grafik")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 975, 26))
        self.menubar.setObjectName("menubar")
        self.menuss = QtWidgets.QMenu(self.menubar)
        self.menuss.setObjectName("menuss")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuss.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnTahmin.setText(_translate("MainWindow", "TAHMİNDE BULUN"))
        self.btnResimSec.setText(_translate("MainWindow", "RESİM SEÇ"))
        self.lbl_photo.setText(_translate("MainWindow", "TextLabel"))
        self.lbl_tahmin_yazi.setText(_translate("MainWindow", "TextLabel"))
        self.lbl_tahmin_grafik.setText(_translate("MainWindow", "TextLabel"))
        self.menuss.setTitle(_translate("MainWindow", "ANA SAYFA"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
