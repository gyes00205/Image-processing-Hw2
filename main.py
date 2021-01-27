import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QInputDialog, QLineEdit, QComboBox
import matplotlib.pyplot as plt
import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics,applications

def DrawContour():
    img1 = cv2.imread('Datasets/Q1_Image/coin01.jpg')
    img2 = cv2.imread('Datasets/Q1_Image/coin02.jpg')
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_g = cv2.GaussianBlur(img1_gray, (11,11), 0)
    img2_g = cv2.GaussianBlur(img2_gray, (11,11), 0)
    img1_edged = cv2.Canny(img1_g, 30, 200)
    img2_edged = cv2.Canny(img2_g, 30, 200)

    cnts1, hierarchy1 = cv2.findContours(img1_edged.copy(),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, hierarchy2 = cv2.findContours(img2_edged.copy(),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img1, cnts1, -1, (0,0,255), 3)
    cv2.drawContours(img2, cnts2, -1, (0,0,255), 3)
    cv2.imshow('1', img1)
    cv2.imshow('2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def CountCoins():
    img1 = cv2.imread('Datasets/Q1_Image/coin01.jpg')
    img2 = cv2.imread('Datasets/Q1_Image/coin02.jpg')
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_g = cv2.GaussianBlur(img1_gray, (11,11), 0)
    img2_g = cv2.GaussianBlur(img2_gray, (11,11), 0)
    img1_edged = cv2.Canny(img1_g, 30, 200)
    img2_edged = cv2.Canny(img2_g, 30, 200)

    cnts1, hierarchy1 = cv2.findContours(img1_edged.copy(),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, hierarchy2 = cv2.findContours(img2_edged.copy(),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_coins1.setText('There are ' +str(len(cnts1))+ ' coins in coin01.jpg')
    label_coins2.setText('There are ' +str(len(cnts2))+ ' coins in coin02.jpg')
    # print(len(cnts1), len(cnts2))

    return

def FindCorners():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('Datasets/Q2_Image/*.bmp')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # print(gray.shape)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (11,8),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (11,8), corners2,ret)
            img = cv2.resize(img, (1024,1024))
            # print(img.shape)
            cv2.imshow('img',img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    global mtx
    global dist
    global rvecs
    global tvecs
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return

def FindIntrinsic(mtx):
    print(mtx)
    return

def FindExtrinsic():
    index = int(combox.currentText())
    # print(index)
    # print(rvecs[0])
    R = cv2.Rodrigues(rvecs[index-1])
    # print(R[0])
    # print(cv2.Rodrigues(rvecs[0])[0])
    # print(tvecs[0])
    print(np.hstack((R[0], tvecs[index-1])))
    return

def FindDistortion():
    print(dist)
    return

def AR():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('Datasets/Q3_Image/*.bmp')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # print(gray.shape)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (11,8),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (11,8), corners2,ret)
            # img = cv2.resize(img, (1024,1024))
            # print(img.shape)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    axis = np.float32([[1,1,0], [5,1,0], [3,5,0], [3,3,-3]])
    index = 0
    for fname in images:
        img = cv2.imread(fname)
        imgpts, jac = cv2.projectPoints(axis, rvecs[index], tvecs[index], mtx, dist)
        imgpts = np.int32(imgpts).reshape(-1,2)
        # print(imgpts)
        for i,j in zip(range(3),[3]*3):
            # print(str(i),str(j))
            cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),3)
        for i,j in zip([0,0,1],[1,2,2]):
            cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),3)
        img = cv2.resize(img, (800,800))
        cv2.imshow('1',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        index += 1


    
    return

def StereoDisparityMap():
    def mouse_handler(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            f = 2826
            B = 178
            c = 123
            if disparity_norm[y][x] + c == 0:
                depth = 0
            else:
                depth = int(f*B/(disparity_norm[y][x]+c))
            # 標記點位置
            drawImg = data['img'].copy()
            cv2.circle(drawImg, (x,y), 1, (0,0,255), 5) 
            cv2.putText(drawImg, 'Disparity: '+str(disparity_norm[y][x])+' pixels', (600,550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(drawImg, 'Depth: '+str(depth)+' mm', (600,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            # 改變顯示 window 的內容
            cv2.imshow("disparity", drawImg)
            
            # 顯示 (x,y) 並儲存到 list中
            print("get points: (x, y) = ({}, {})".format(x, y))
            data['points'].append((x,y))

    def get_points(im):
        # 建立 data dict, img:存放圖片, points:存放點
        data = {}
        data['img'] = im.copy()
        data['points'] = []
            
        # 顯示圖片在 window 中
        cv2.imshow('disparity',im)
        
        # 利用滑鼠回傳值，資料皆保存於 data dict中
        cv2.setMouseCallback("disparity", mouse_handler, data)
        
        # 等待按下任意鍵，藉由 OpenCV 內建函數釋放資源
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        # 回傳點 list
        return data['points']
    
    
    imgL = cv2.imread('Datasets/Q4_Image/imgL.png',0)
    imgR = cv2.imread('Datasets/Q4_Image/imgR.png',0)
    # print(imgL.shape)
    
    stereo = cv2.StereoBM_create(numDisparities=16*21, blockSize=19)
    disparity = stereo.compute(imgL,imgR)
    # print(np.max(disparity))
    # print(np.min(disparity))
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_norm = cv2.resize(disparity_norm, (disparity_norm.shape[1]//3, disparity_norm.shape[0]//3))
    disparity_vis = cv2.cvtColor(disparity_norm, cv2.COLOR_GRAY2BGR)
    # disparity = cv2.resize(disparity, (disparity.shape[1]//3, disparity.shape[0]//3))
    
    points = get_points(disparity_vis)
    # print(points)
    return

def Showboard():
    img = cv2.imread('tensorboard.png')
    cv2.imshow('tensorboard', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def PredictTest():
    testIndex = int(testInput.text()) - 1
    originImg = cv2.imread(img_list[testIndex])[:,:,::-1]
    img = cv2.resize(originImg.copy(), (256,256)) / 255.0
    img = img.reshape((1,256,256,3))
    img_pred = new_model.predict(img)[0]
    ans = 'cat' if img_pred[0]>img_pred[1] else 'dog'
    print(img_pred)
    plt.imshow(originImg)
    plt.title('class:'+ans)
    plt.show()
    return

def Comparison():
    img = cv2.imread('before_after_resize.png')
    cv2.imshow('comparison', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

solve_cudnn_error()
new_model = keras.models.load_model('cat_dog_model')
img_list = glob.glob('test/*/*.jpg')
mtx, dist, rvecs, tvecs = None, None, None, None
# disparity_norm = None
app = QApplication([])
app.setApplicationName('2020 Opencvdl HW2')
app.setStyle('Fusion')
window = QWidget()
window.setGeometry(500,100,800,500)

label1 = QLabel(window)
label1.setGeometry(10,10,110,60)
label1.setText('1.Find contour')

btn11 = QPushButton(window)
btn11.setGeometry(10,70,150,30)
btn11.setText('1.1 Draw contour')
btn11.clicked.connect(lambda : DrawContour())

btn12 = QPushButton(window)
btn12.setGeometry(10,110,150,30)
btn12.setText('1.2 Count coins')
btn12.clicked.connect(lambda : CountCoins())

label_coins1 = QLabel(window)
label_coins1.setGeometry(10,150,200,20)
label_coins1.setText('There are __ coins in coin01.jpg')
label_coins2 = QLabel(window)
label_coins2.setGeometry(10,170,200,20)
label_coins2.setText('There are __ coins in coin02.jpg')

label2 = QLabel(window)
label2.setGeometry(10,200,110,60)
label2.setText('2.Calibration')

btn21 = QPushButton(window)
btn21.setGeometry(10,260,150,30)
btn21.setText('2.1 Find Corners')
btn21.clicked.connect(lambda :FindCorners())

btn22 = QPushButton(window)
btn22.setGeometry(10,300,150,30)
btn22.setText('2.2 Find Intrinsic')
btn22.clicked.connect(lambda : FindIntrinsic(mtx))

label23 = QLabel(window)
label23.setGeometry(230,240,150,80)
label23.setText('2.3 Find Extrinsic\n      Select image')

combox = QComboBox(window)
combox.setGeometry(230,310,150,30)
combox.addItems(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])

btn23 = QPushButton(window)
btn23.setGeometry(230,350,150,30)
btn23.setText('2.3 Find Extrinsic')
btn23.clicked.connect(lambda : FindExtrinsic())

btn24 = QPushButton(window)
btn24.setGeometry(10,340,150,30)
btn24.setText('2.4 Find Distortion')
btn24.clicked.connect(lambda : FindDistortion())

label3 = QLabel(window)
label3.setGeometry(230,10,150,60)
label3.setText('3.Augmented Reality')

btn31 = QPushButton(window)
btn31.setGeometry(230,80,150,30)
btn31.setText('3.1 Augmented Reality')
btn31.clicked.connect(lambda : AR())

label4 = QLabel(window)
label4.setGeometry(230,120,150,60)
label4.setText('4.Stereo Disparity Map')

btn41 = QPushButton(window)
btn41.setGeometry(230,190,150,30)
btn41.setText('4.1 Stereo Disparity Map')
btn41.clicked.connect(lambda : StereoDisparityMap())

label5 = QLabel(window)
label5.setGeometry(450,10,150,60)
label5.setText('5.ResNet50')

btn51 = QPushButton(window)
btn51.setGeometry(450,80,150,30)
btn51.setText('5.1 Show tensorboard')
btn51.clicked.connect(lambda : Showboard())

label52 = QLabel(window)
label52.setGeometry(450,120,150,60)
label52.setText('Input: 1~2000')

testInput = QLineEdit(window)
testInput.setGeometry(450,160,150,30)

btn52 = QPushButton(window)
btn52.setGeometry(450,200,150,30)
btn52.setText('5.2 Predict Test')
btn52.clicked.connect(lambda : PredictTest())

btn53 = QPushButton(window)
btn53.setGeometry(450,240,150,30)
btn53.setText('5.3 Comparison')
btn53.clicked.connect(lambda : Comparison())

window.show()
app.exec_()
