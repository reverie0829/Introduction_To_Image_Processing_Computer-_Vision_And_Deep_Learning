import sys
import cv2
import numpy as np
from Ui_Hw1 import Ui_mainWindow
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication

class MainWindow(QMainWindow, Ui_mainWindow):

    Gaussian_3_1 = None
    Sobel_X_3_2 = None
    Sobel_Y_3_3 = None
    label_8 = None

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    def onBindingUI(self):
        self.pushButton.clicked.connect(self.on_btn1_1_click)
        self.pushButton_2.clicked.connect(self.on_btn1_2_click)
        self.pushButton_3.clicked.connect(self.on_btn1_3_click)
        self.pushButton_4.clicked.connect(self.on_btn1_4_click)
        self.pushButton_5.clicked.connect(self.on_btn2_1_click)
        self.pushButton_6.clicked.connect(self.on_btn2_2_click)
        self.pushButton_7.clicked.connect(self.on_btn2_3_click)
        self.pushButton_8.clicked.connect(self.on_btn3_1_click)
        self.pushButton_9.clicked.connect(self.on_btn3_2_click)
        self.pushButton_10.clicked.connect(self.on_btn3_3_click)
        self.pushButton_11.clicked.connect(self.on_btn3_4_click)
        self.pushButton_12.clicked.connect(self.on_btn4_click)

    def on_btn1_1_click(self):
        image = cv2.imread('Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
        
        print(f'Height = {image.shape[1]}')
        print(f'Width  = {image.shape[0]}')
        #image = cv2.resize(image,(720,480))       
        cv2.imshow('1.1 Load Image File',image)
        # b,g,r = cv2.split(image)
        # img_rgb = cv2.merge([r,g,b])
        # plt.imshow(img_rgb)
        # plt.show()

    def on_btn1_2_click(self):
        image = cv2.imread('Dataset_opencvdl/Q1_Image/Flower.jpg')

        # set green and red channels to 0
        b = image.copy()        
        b[:, :, 1] = b[:, :, 2] = 0        
        # set blue and red channels to 0
        g = image.copy()        
        g[:, :, 0] = g[:, :, 2] = 0        
        # set blue and green channels to 0
        r = image.copy()        
        r[:, :, 0] = r[:, :, 1] = 0        
        # RGB - Blue
        cv2.imshow('B-RGB', b)
        # RGB - Green
        cv2.imshow('G-RGB', g)
        # RGB - Red
        cv2.imshow('R-RGB', r)

    def on_btn1_3_click(self):
        image = cv2.imread('Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
        image = cv2.resize(image,(720,480)) 
        image_flip = cv2.flip(image, 1)

        cv2.imshow('Original Image',image)
        cv2.imshow('Result Image',image_flip)

    def on_btn1_4_click(self):
        def blending(x):
            p = cv2.getTrackbarPos('BLEND', '1-4 Blending')/255
            cv2.imshow('1-4 Blending', cv2.addWeighted(image, p, image_flip, 1-p, 0.0))
        image = cv2.imread('Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
        image = cv2.resize(image,(720,480)) 
        image_flip = cv2.flip(image, 1)
        cv2.namedWindow('1-4 Blending')
        cv2.imshow('1-4 Blending', cv2.addWeighted(image, 0, image_flip, 1, 0.0))
        cv2.createTrackbar('BLEND', '1-4 Blending', 0, 255, blending)

    def on_btn2_1_click(self):
        image = cv2.imread('Dataset_opencvdl\Q2_Image\Cat.png')
        median = cv2.medianBlur(image,7)
        cv2.imshow('median',median)

    def on_btn2_2_click(self):
        image = cv2.imread('Dataset_opencvdl\Q2_Image\Cat.png')
        Gaussian = cv2.GaussianBlur(image, (3, 3), 0)
        cv2.imshow('Gaussian',Gaussian)

    def on_btn2_3_click(self):
        image = cv2.imread('Dataset_opencvdl\Q2_Image\Cat.png')
        Bilateral = cv2.bilateralFilter(image, 9, 90, 90)
        cv2.imshow('Bilateral',Bilateral)

    def on_btn3_1_click(self):
        image = cv2.imread('Dataset_opencvdl\Q3_Image\Chihiro.jpg', cv2.IMREAD_GRAYSCALE)
        #3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = (1/2*np.pi)*(np.exp(-(x**2+y**2)))
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        #print(gaussian_kernel)
        
        self.Gaussian_3_1 = _conv_2d_3x3(image, gaussian_kernel)
        # Gaussian = abs(self.Gaussian_3_1)
        # Gaussian = (Gaussian / np.max(Gaussian) * 255).astype(np.uint8)
        Gaussian = self.Gaussian_3_1.astype(np.uint8)
        cv2.imshow('Gaussian', Gaussian)
        self.label_8.clear()
    
    def on_btn3_2_click(self):
        if self.Gaussian_3_1 is None :
            print('Please run "3.1 Gaussian Blur" first.')
            self.label_8.setText('Warning!! Please run "3.1 Gaussian Blur" first.')
            self.label_8.setStyleSheet("color: rgb(255, 0, 0);") 
        else :
            image = self.Gaussian_3_1
            k = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
            self.Sobel_X_3_2 = _conv_2d_3x3(image, k)
            gx = abs(self.Sobel_X_3_2)
            gx = (gx / np.max(gx) * 255).astype(np.uint8)

            cv2.imshow('4.2 Sobel X', gx)
            self.label_8.clear()

    def on_btn3_3_click(self):
        if self.Gaussian_3_1 is None :
            print('Please run "3.1 Gaussian Blur" first.')
            self.label_8.setText('Warning!! Please run "3.1 Gaussian Blur" first.')
            self.label_8.setStyleSheet("color: rgb(255, 0, 0);") 
        else :
            image = self.Gaussian_3_1
            k = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]], dtype=np.float32)
            self.Sobel_Y_3_3 = _conv_2d_3x3(image, k)
            gy = abs(self.Sobel_Y_3_3)
            gy = (gy / np.max(gy) * 255).astype(np.uint8)

            cv2.imshow('4.3 Sobel Y', gy)
            self.label_8.clear()

    def on_btn3_4_click(self):
        if self.Sobel_X_3_2 is None or self.Sobel_Y_3_3 is None:
            print('Please run "3.1 ~ 3.3" first.')
            self.label_8.setText('Warning!! Please run "3.1 ~ 3.3" first.')
            self.label_8.setStyleSheet("color: rgb(255, 0, 0);")         
        else :
            image = np.sqrt(np.power((self.Sobel_X_3_2),2) + np.power((self.Sobel_Y_3_3),2))
            image = (image / np.max(image) * 255).astype(np.uint8)

            cv2.imshow('4.4 Magnitude', image)
            self.label_8.clear()

    def on_btn4_click(self):
        image  = cv2.imread('Dataset_opencvdl\Q4_Image\Parrot.png')

        angle = self.lineEdit.text()
        scale = self.lineEdit_2.text()
        tx    = self.lineEdit_3.text()
        ty    = self.lineEdit_4.text()
        if not len(angle) or not len(scale) or not len(tx) or not len(ty):
            return
        
        M = np.array([[1, 0, float(tx)], [0, 1, float(ty)]])
        _img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        M = cv2.getRotationMatrix2D((160+float(tx), 84+float(ty)), float(angle), float(scale))
        _img = cv2.warpAffine(_img, M, (image.shape[1], image.shape[0]))

        cv2.imshow('Original Image', image)
        cv2.imshow('Result Image',_img)


def _conv_2d_3x3(image, kernel):
    res = np.zeros_like(image).astype(np.float32)
    image = np.pad(image, 1).astype(np.float32)

    for r in range(1, image.shape[0] - 1):
        for c in range(1, image.shape[1] - 1):
            res[r-1, c-1] = np.sum(kernel * image[r-1:r+2, c-1:c+2])
    return res


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())