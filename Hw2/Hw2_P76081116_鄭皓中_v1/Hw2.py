import os
import sys
import cv2
import glob
import numpy as np
from PIL import Image
from Hw2_ui import Ui_MainWindow
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication
from keras import layers, optimizers, models
from keras.applications.resnet50 import ResNet50
from keras.layers import *    
from keras.models import Model
from keras.optimizers import SGD ,Adam
from keras.callbacks import CSVLogger




class MainWindow(QMainWindow, Ui_MainWindow):

    label = 'There are __ coins in coin01.jpg'
    label_2 = 'There are __ coins in coin02.jpg'

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None


    def onBindingUI(self):
        self.pushButton.clicked.connect(self.on_btn1_1_click)
        self.pushButton_2.clicked.connect(self.on_btn1_2_click)
        self.pushButton_3.clicked.connect(self.on_btn2_1_click)
        self.pushButton_4.clicked.connect(self.on_btn2_2_click)
        self.pushButton_5.clicked.connect(self.on_btn2_4_click)
        self.pushButton_6.clicked.connect(self.on_btn2_3_click)
        self.pushButton_7.clicked.connect(self.on_btn3_1_click)
        self.pushButton_8.clicked.connect(self.on_btn4_1_click)
        self.pushButton_9.clicked.connect(self.on_btn5_1_click)
        self.pushButton_10.clicked.connect(self.on_btn5_2_click)
        self.pushButton_11.clicked.connect(self.on_btn5_3_click)
        self.pushButton_12.clicked.connect(self.on_btn5_4_click)

    def on_btn1_1_click(self):
        # print('1.1 Draw Contour')
        image1 = cv2.imread("Datasets/Q1_Image/coin01.jpg")
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        blurred1 = cv2.GaussianBlur(gray1, (11, 11), 0)
        edged1 = cv2.Canny(blurred1, 30, 150)
        cnts1 = cv2.findContours(
            edged1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        # print(" {} coins in the image1".format(len(cnts1)))
        coins1 = image1.copy()
        cv2.drawContours(coins1, cnts1, -1, (0, 0, 255), 2)

        image2 = cv2.imread("Datasets/Q1_Image/coin02.jpg")
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        blurred2 = cv2.GaussianBlur(gray2, (11, 11), 0)
        edged2 = cv2.Canny(blurred2, 30, 150)
        cnts2 = cv2.findContours(
            edged2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        # print(" {} coins in the image2".format(len(cnts2)))
        coins2 = image2.copy()
        cv2.drawContours(coins2, cnts2, -1, (0, 0, 255), 2)

        result = cv2.hconcat([coins1, coins2])
        cv2.imshow('1.1 Draw Contour', result)

    def on_btn1_2_click(self):
        # print('1.2 Count coins')
        image1 = cv2.imread("Datasets/Q1_Image/coin01.jpg")
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        blurred1 = cv2.GaussianBlur(gray1, (11, 11), 0)
        edged1 = cv2.Canny(blurred1, 30, 150)
        cnts1 = cv2.findContours(
            edged1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        # print(" {} coins in the image1".format(len(cnts1)))
        self.label.setText(
            'There are {} coins in coin01.jpg'.format(len(cnts1)))

        image2 = cv2.imread("Datasets/Q1_Image/coin02.jpg")
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        blurred2 = cv2.GaussianBlur(gray2, (11, 11), 0)
        edged2 = cv2.Canny(blurred2, 30, 150)
        cnts2 = cv2.findContours(
            edged2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        # print(" {} coins in the image2".format(len(cnts2)))
        self.label_2.setText(
            'There are {} coins in coin02.jpg'.format(len(cnts2)))

    def on_btn2_1_click(self):
        for i in range(15):
            filename = 'Datasets/Q2_Image/' + str(i + 1) + '.bmp'
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (11, 8), None) #行列數12*9，因為是算內部的角點所以長寬都要減一(12-1)*(9-1)
            if found == True:
                cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
                cv2.drawChessboardCorners(img, (11,8), corners, found)
                cv2.imshow(filename, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn2_2_click(self):
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        for i in range(15):
            filename = 'Datasets/Q2_Image/' + str(i + 1) + '.bmp'
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if found == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        #已獲取三維點和對應影像的二維點，返回標定結果
        #我們要使用的函式是 cv2.calibrateCamera()。它會返回攝像機矩陣，畸變係數，旋轉和變換向量等。
        found, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        print(self.mtx)

    def on_btn2_4_click(self):
        if self.dist is None :
            print('Please run 2.2 Find Intrinsic first')
        else :
            print(self.dist)

    def on_btn2_3_click(self):
        if self.rvecs is None :
            print('Please run 2.2 Find Intrinsic first')
        else :
            j = self.comboBox.currentIndex()
            # Return rotation matrix & Jacobian
            dst, _ = cv2.Rodrigues(self.rvecs[j])
            dst = dst.tolist()
            for i in range(len(dst)):
                dst[i].append(self.tvecs[j][i][0])
            print(np.array(dst))

    def on_btn3_1_click(self):
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []

        mtx = []
        dist = []
        for i in range(5):
            filename = 'Datasets/Q3_Image/' + str(i + 1) + '.bmp'
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if found == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        found, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        axis = np.float32(
            [[3, 5, 0], [5, 1, 0], [1, 1, 0], [3, 3, -3]])
        
        all_file = []
        for i in range(5):
            all_file.append(
                'Datasets/Q3_Image/' + str((i+1)) + '.bmp')
        img_array = []
        h = 0
        for filename in all_file:
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if found == True:
                # To project 3D points to image plane
                imgpts, jac = cv2.projectPoints(
                    axis, rvecs[h], tvecs[h], mtx, dist)
                # print(rvecs[0])
                # print(tvecs[0])
                # To draw objects
                pt1 = tuple(imgpts[0].ravel())
                pt2 = tuple(imgpts[1].ravel())
                pt3 = tuple(imgpts[2].ravel())
                pt4 = tuple(imgpts[3].ravel())

                img = cv2.line(img, pt1, pt2, (0, 0, 255), 10)
                img = cv2.line(img, pt2, pt3, (0, 0, 255), 10)
                img = cv2.line(img, pt3, pt1, (0, 0, 255), 10)
                img = cv2.line(img, pt1, pt4, (0, 0, 255), 10)
                img = cv2.line(img, pt2, pt4, (0, 0, 255), 10)
                img = cv2.line(img, pt3, pt4, (0, 0, 255), 10)

                img_array.append(img)
                height, width = img.shape[:2]
                size = (width, height)
            h += 1

        out = cv2.VideoWriter('Q_3.avi', cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 2, size)  # 2 is fps
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        cap = cv2.VideoCapture('Q_3.avi')
        if (cap.isOpened() == False):
            print('gg')
        while(cap.isOpened()):
            found, frame = cap.read()
            if found == True:
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Frame", 1024, 1024)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def on_btn4_1_click(self):
        def on_mouse_display_depth_value(event, x, y, flags, params):
        # when the left mouse button has been clicked
            if (event == cv2.EVENT_LBUTTONDOWN):
                f, B = params
                if (disparity[y,x] > 0):
                    depth = (f * B) / (disparity[y,x]+123)
                    Disparity = 'Disparity: '+str(int(disparity[y,x]))+' pixels'
                    Depth = 'Depth: '+str(int(depth))+' mm'
                    cv2.rectangle(disparity,(1050,860),(1410,960),(255,255,255),-1)
                    cv2.putText(disparity, Disparity, (1070, 900), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0) ,2, cv2.LINE_AA)
                    cv2.putText(disparity, Depth, (1070, 950), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0) ,2, cv2.LINE_AA)
                    cv2.imshow("disparity", disparity)
                else:
                    print('請選別的點')
                    other = 'Please choose other '
                    other2 = 'point.'
                    cv2.rectangle(disparity,(1050,860),(1410,960),(255,255,255),-1)
                    cv2.putText(disparity, other, (1070, 900), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0) ,2, cv2.LINE_AA)
                    cv2.putText(disparity, other2, (1070, 950), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0) ,2, cv2.LINE_AA)
                    cv2.imshow("disparity", disparity)

        
        imgL = cv2.imread('Datasets/Q4_Image/imgL.png', 0)
        imgR = cv2.imread('Datasets/Q4_Image/imgR.png', 0)
        
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL, imgR)#.astype(np.uint8)
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disparity = cv2.resize(disparity,(1410,960))

        cv2.imshow('disparity',disparity)
        cv2.setMouseCallback('disparity',on_mouse_display_depth_value, (2826,178))

    def on_btn5_1_click(self):
        #讀取文件夾PetImages下的2000張圖片，圖片為彩色圖，所以為3通道，
        #如果是將彩色圖作為輸入,圖像大小224*224

        EPOCHS = 5
        BATCH_SIZE = 32

        Cat_Vs_Dogs = np.load('No5_dataset/Cat_Vs_Dogs.npz')
        data = Cat_Vs_Dogs['data']
        label = Cat_Vs_Dogs['label']

        conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        conv_base.trainable = False

        model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
        #logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        #model.summary()

        #Train model
        model.fit(data, label, epochs=EPOCHS, batch_size=BATCH_SIZE)#,callbacks=[tensorboard_callback])#, validation_split=0.2)
        model.save("resnet50.h5")

    def on_btn5_2_click(self):
        tensorboard = cv2.imread('No5_dataset/screenshot_of_TensorBoard.png')
        cv2.imshow('5.2 TensorBoard',tensorboard)

    def on_btn5_3_click(self):
        tmp = np.random.randint(2000)
        if tmp < 1000:
            catimgs = sorted(os.listdir('No5_dataset/PetImages/Cat/'))
            img = cv2.imread('No5_dataset/PetImages/Cat/'+catimgs[tmp+1000])
            img = cv2.resize(img,(224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title('Class:cat')
            plt.show()

        else :
            dogimgs = sorted(os.listdir('No5_dataset/PetImages/Dog/'))
            img = cv2.imread('No5_dataset/PetImages/Dog/'+dogimgs[tmp])
            img = cv2.resize(img,(224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title('Class:dog')
            plt.show()

    def on_btn5_4_click(self):
        # acc = [65.3, 69]
        # method=['Before Resize', 'After Resize']
        # x=np.arange(len(method))
        # plt.bar(x, acc, tick_label=method)

        # plt.title('Resize augmentation comparison')
        # plt.ylim(64,70)
        # plt.show()

        def random_resize_data():                   
            catimgs = sorted(os.listdir('No5_dataset/PetImages/Cat/'))
            catnum = len(catimgs)        
            for i in range(catnum):     
                cat = cv2.imread('No5_dataset/PetImages/Cat/'+catimgs[i])
                cat = cv2.resize(cat,(np.random.randint(1000),np.random.randint(1000)))
                cat = cv2.imwrite('No5_dataset/PetImages/Cat/'+catimgs[i],cat)

            dogimgs = sorted(os.listdir('No5_dataset/PetImages/Dog/'))
            dognum = len(dogimgs)
            for i in range(dognum):
                dog = cv2.imread('No5_dataset/PetImages/Dog/'+dogimgs[i])
                dog = cv2.resize(dog,(np.random.randint(1000),np.random.randint(1000)))
                dog = cv2.imwrite('No5_dataset/PetImages/Dog/'+dogimgs[i],dog)


        img = cv2.imread('No5_dataset/Try_to_resize.png')
        cv2.imshow('5.4 Try to resize',img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
