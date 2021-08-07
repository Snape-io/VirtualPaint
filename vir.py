import cv2 as cv
import numpy as np
import time
import os
import picker as p
folderPath ="VirPaint/top"
L1=os.listdir(folderPath)

overlayL=[]
for impath in L1:
    top = cv.imread(f'{folderPath}/{impath}')
    overlayL.append(top)
 
header = overlayL[1]
Dcolor=(0,0,255) 
Font=15
Xp,Yp=0,0 
frameW = 1280
frameH = 728
cam = cv.VideoCapture(0, cv.CAP_DSHOW)
cam.set(3, frameW)
cam.set(4,frameH)
cam.set(10,200)
drawingB= np.zeros((720,1280,3),np.uint8)
detector =p.handDetector(detectionCon=0.85)
while True:
    success,img = cam.read()
    img = cv.flip(img,1)
    img[0:125,0:1280] = header
    img = detector.findHands(img)
    lmL = detector.findPosition(img, draw=False)
    if len(lmL)!=0:
        
        X,Y =lmL[8][1:]
        x,y =lmL[12][1:]
        F=detector.fingersUp()
        
        if F[1] and F[2]:
            Xp,Yp=0,0
            cv.rectangle(img,(X,Y-15),(x,y+15),Dcolor,cv.FILLED)
            if Y<150:
                if 245<X<320:
                    header=overlayL[1]
                    Dcolor =(0,0,255)
                elif 355<X<430:
                    header=overlayL[2]
                    Dcolor=(205,0,0)    
                elif 470<X<545:
                    header=overlayL[3]
                    Dcolor=(0,0,0)
                elif 580<X<655:
                    header=overlayL[4]
                    Dcolor=(0,255,255)
                elif 688<X<761:
                    header=overlayL[5]
                    Dcolor=(0,165,255)

                elif 790<X<869:
                    header=overlayL[6]
                    Dcolor=(255,255,255)
                elif 944<X<1069:
                    header=overlayL[7]
                    Dcolor=(1,0,0)
                                             
            
        if F[1] and F[2]== False:
            cv.circle(img,(X,Y),10,Dcolor,cv.FILLED)
            if Xp==0 and Yp==0:
                Xp,Yp=X,Y
            cv.line(img,(Xp,Yp),(X,Y),Dcolor,Font)
            cv.line(drawingB,(Xp,Yp),(X,Y),Dcolor,Font)
            Xp,Yp = X,Y
            if Dcolor==(1,0,0):
                cv.line(img,(Xp,Yp),(X,Y),Dcolor,40)
                cv.line(drawingB,(Xp,Yp),(X,Y),Dcolor,40)
    imgG=cv.cvtColor(drawingB,cv.COLOR_BGR2GRAY) 
    _,imgI=cv.threshold(imgG,50,255,cv.THRESH_BINARY_INV)
    imgI=cv.cvtColor(imgI,cv.COLOR_GRAY2BGR) 
    img =cv.bitwise_and(img,imgI)
    img=cv.bitwise_or(img,drawingB)
    img =cv.addWeighted(img,0.5,drawingB,0.5,0)       
    cv.imshow("webcam",img)
    
    if cv.waitKey(1) & 0xFF == ord('s'):
        break
cv.destroyAllWindows()


