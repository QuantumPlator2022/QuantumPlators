import cv2
import numpy as np
import PIL as Image
import os
from os import listdir


############################################################################################

frameCarNr = frameVehNr = frameNr = (x for x in range(1000))


"""############################################################################################"""

folder_dir1 = "C:/Users/Sarveshwar/Desktop/output/Pedestrians"
T_cascade_src = r"C:\Users\Sarveshwar\Downloads\QAlgo\two_wheeler.xml"
#T_video_src = r"C:\Users\Sarveshwar\Downloads\two_wheeler2.mp4"
folder_dir2 = "C:/Users/Sarveshwar/Desktop/output/Two_Wheeler"
C_cascade_src = r"C:\Users\Sarveshwar\Downloads\QAlgo\cars.xml"
#C_video_src = r"C:\Users\Sarveshwar\Downloads\Main Project_Main Project_Car Detection_video.avi"
folder_dir3 = "C:/Users/Sarveshwar/Desktop/output/Cars"
############################################################################################
def Ped(video_source : "str"):
    full_body = cv2.CascadeClassifier(r'C:\Users\Sarveshwar\Downloads\QAlgo\pedestrians.xml')
    cap = cv2.VideoCapture(video_source)
    global gray, framePed
    while cap.isOpened():
        framePed = cap.read()
        gray = cv2.cvtColor(framePed, cv2.COLOR_BGR2GRAY)
        body = full_body.detectMultiScale(gray, 1.2, 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for (x,y,w,h) in body: 
            cv2.putText(framePed, 'Person', (x, y-5), font, 1, (255,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(framePed,(x,y),(x+w,y+h),(255,255,0),2)
            cv2.imshow('Body Detection',framePed)
        
            cv2.imwrite(f'C:/Users/Sarveshwar/Desktop/output/Pedestrians/frame_{frameNr}.jpg', framePed)
        #"C:\Users\Sarveshwar\Desktop\output"
        k = cv2.waitKey(1)
        if k == 40:
            break
        
    cap.release() 
    cv2.destroyAllWindows()

"""###################################################################################################"""
def PC():

    for images in os.listdir(folder_dir1):
        if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
            image_arr2 = np.array(images)
            cnt = 0
            for (x,y,w,h) in images:
                cv2.rectangle(image_arr2,(x,y),(x+w,y+h),(255,0,0),2)
                cnt += 1
                #print(cnt, " people found")
                Image.fromarray(image_arr2)
            #print(images)
    return 1


#######################################################################################
"""**Vehicle(2 wheeler)**"""

def Twheeler(T_video_src: str):
    cap = cv2.VideoCapture(T_video_src)
    car_cascade = cv2.CascadeClassifier(T_cascade_src)
    while True:
        ret, img = cap.read()
        if (type(img) == type(None)):
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        veh = car_cascade.detectMultiScale(gray,1.01, 1)
        for (x,y,w,h) in veh:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),2)
            cv2.imshow('2 wheeler detection', img)
            cv2.imwrite(f'C:/Users/Sarveshwar/Desktop/output/Two_Wheeler/frame_{frameVehNr}.jpg', img)
        if cv2.waitKey(33) == 27:
            break
    cv2.destroyAllWindows()
    
"""##################################################################################################"""

def TWC():
    for vehicles in os.listdir(folder_dir2):
        if (vehicles.endswith(".png") or vehicles.endswith(".jpg") or vehicles.endswith(".jpeg")):
            image_arrVeh = np.array(vehicles)
            cntVeh = 0
            for (x,y,w,h) in vehicles:
                cv2.rectangle(image_arrVeh,(x,y),(x+w,y+h),(255,0,0),2)
                cntVeh += 1
                #print(cntVeh, "2 wheelers found")
                Image.fromarray(image_arrVeh)
            #print(vehicles)
    return 1
########################################################################################
"""**Cars**"""
def cars(C_video_src : "str"):
    cap = cv2.VideoCapture(C_video_src)
    car_cascade = cv2.CascadeClassifier(C_cascade_src)
    while True:
        ret, imgCars = cap.read()
        if (type(imgCars) == type(None)):
            break
        gray = cv2.cvtColor(imgCars, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 2)
        for (x,y,w,h) in cars:
            cv2.rectangle(imgCars,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.imshow('video', imgCars)
            cv2.imwrite(f'C:/Users/Sarveshwar/Desktop/output/Cars/frame_{frameCarNr}.jpg', imgCars)
    
        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()
    
"""##################################################################################################"""

def CCo():
    for cars in os.listdir(folder_dir3):
        if (cars.endswith(".png") or cars.endswith(".jpg") or cars.endswith(".jpeg")):
            image_arrCar = np.array(cars)
            global cntCar
            cntCar = 0
            for (x,y,w,h) in cars:
                cv2.rectangle(image_arrCar,(x,y),(x+w,y+h),(255,0,0),2)
                cntCar += 1
                print(cntCar, "Cars found")
                Image.fromarray(image_arrCar)
            print(cars)
    return 1
        
##############################################################################################

def FRML():
    Main = list()
    dm = list()
    ML = list()
    for i in range(4):
        a, b, c = input("Input the path to pedestrian video source file:- "), input("Input the path to Two wheeler video source file:- "), input("Input the path to Car video source file:- ")
        Ped(a)
        Twheeler(b)
        cars(c)
        O = [float(CCo()+ TWC()), PC()]
        ML.append(O)
    Main.append(ML)
    dm.append(ML)
    global x, y
    x = float(input("Enter the X-Coordinate of Node:- ")), 
    y = float(input("Enter the Y-Coordiante of Node:- "))
    M = [x,y]
    Main.append(M)
    dm.append(M)
    t = int(input("Numbers of Nodes connected:- "))
    OP = list()
    for j in range(int(t)):
        N = str(input("Enter the Node Name:- "))
        OP.append(N)
    for i in range(4):
        Main[0][i].append(OP[i])
    A = Main[0]
    dm.append(A)
    return dm