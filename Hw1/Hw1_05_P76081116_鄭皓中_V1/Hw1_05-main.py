import sys
import cv2
import random
import numpy as np
import pandas as pd
from Ui_no5_ui import Ui_MainWindow
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication

import keras
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten

_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

class MainWindow(QMainWindow, Ui_MainWindow):

    label_11 = None
    x_train = None
    y_train = None        
    x_test = None        
    y_test = None

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    def onBindingUI(self):
        self.pushButton_13.clicked.connect(self.on_btn5_1_click)
        self.pushButton_14.clicked.connect(self.on_btn5_2_click)
        self.pushButton_15.clicked.connect(self.on_btn5_3_click)
        self.pushButton_16.clicked.connect(self.on_btn5_4_click)
        self.pushButton_17.clicked.connect(self.on_btn5_5_click)

    def on_btn5_1_click(self):
        self.label_11.clear()
        #Import dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)
        #print(self.x_train ,self.x_test ,'分隔', self.y_train , self.y_test)

        #plot_images_labels
        fig = plt.figure('5.1 Show Train Images',figsize=(10,5))
        fig.subplots_adjust(hspace=0.0,wspace=0.4)

        for i in range(0, 10):                                                      #依序顯示 num 個子圖
            ax = fig.add_subplot(2, 5, i+1)                                         #建立 2*5 個子圖中的第 i+1 個
            temp = random.randint(0,9999)
            x_train_resize=cv2.resize(self.x_train[temp], (128, 128))
            ax.imshow(np.uint8(x_train_resize))
            ax.set_title(_LABELS[list(self.y_train[temp]).index(1)],fontsize=10)    #設定標題
            ax.set_xticks([]);                                                      #不顯示 x 軸刻度
            ax.set_yticks([]);                                                      #不顯示 y 軸刻度
        plt.show() 


    def on_btn5_2_click(self):
        print('hyperparameters:')
        print('batch size:', 32)
        print('learning rate:', 0.001)
        print('optimizer:','SGD')

    def on_btn5_3_click(self):
        input_shape = (32, 32, 3)
        model = Sequential()
        model.add(VGG16(weights=None, include_top=False,input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.summary()

    def on_btn5_4_click(self):
        Accurancy  = cv2.imread('Accurancy.png')
        loss = cv2.imread('loss.png')
        image = cv2.vconcat([Accurancy,loss])
        cv2.imshow('5.4 Show Accuracy', image)

    def on_btn5_5_click(self):
        self.label_11.clear()
        if self.x_train is None or self.y_train is None or self.x_test is None or  self.y_test is None :
            print('Please run "5.1 Show Train Images" first.')
            self.label_11.setText('Warning!! Please run "5.1 Show Train Images" first.')
            self.label_11.setStyleSheet("color: rgb(255, 0, 0);")
            #print(self.x_train ,self.y_train ,self.x_test , self.y_test)
        elif not len(self.lineEdit_5.text()) :
            print('Please enter (0~9999) first.')
            self.label_11.setText('Warning!! Please enter (0~9999) first.')
            self.label_11.setStyleSheet("color: rgb(255, 0, 0);")
            
        else :
            INDEX = int(self.lineEdit_5.text())
            
            model = load_model("cifar10_vgg16.h5")
            probabilities = model.predict(self.x_test)
            
            plt.figure('Estimation result',figsize=(10, 6), dpi=80)
            plt.bar(["airplain", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], list(probabilities[INDEX]), 0.5)
            plt.title('Estimation result')
            x_test = self.x_test.astype(np.uint8)
            plt.figure('image')
            plt.imshow(cv2.resize(x_test[INDEX], (128, 128)))
            title = _LABELS[list(self.y_test[INDEX]).index(1)]
            plt.title(title, fontsize=12)
            plt.xticks([])
            plt.yticks([])        
            plt.show()   


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())