import config
import cv2
import os
from imutils import paths
from multiprocessing import Pool

imagePaths = list(paths.list_images(config.DATASET_FOLDER))

def crop(index):
    global imagePaths
    imagePath = imagePaths[index]
    # if os.path.isfile(imagePath.replace("test","testAligned")):
    #     return
    try:
        image = cv2.imread(imagePath)
        imgHeight = image.shape[0]
        imgWidth = image.shape[1]
        textPath = imagePath.replace(".jpg","_BB.txt")
        f = open(textPath,"r")
        (x, y, w, h, confidence) = [float(i) for i in f.readline()[:-2].split(" ")]

        xRatio = imgWidth/224
        yRatio = imgHeight/224
        width = xRatio*w
        height = yRatio*h
        xMid = xRatio * (x + w/2)
        yMid = yRatio * (y + h/2)
        if width>height:        
            xl = int(xMid - width/2)
            xr = int(xMid + width/2)
            yu = int(yMid - width/2)
            yd = int(yMid + width/2)

        else:
            xl = int(xMid - height/2)
            xr = int(xMid + height/2)
            yu = int(yMid - height/2)
            yd = int(yMid + height/2)
        
        lackOfyu = lackOfyd = lackOfxl = lackOfxr = 0

        if yu<0:
            lackOfyu = 0-yu
            yu=0
        if yd>=imgHeight:
            lackOfyd = yd-imgHeight+1
            yd = imgHeight-1
        if xl<0:
            lackOfxl = 0-xl
            xl = 0
        if xr>=imgWidth:
            lackOfxr = xr-imgWidth+1
            xr = imgWidth-1

        cropImg = image[yu:yd, xl:xr]
        BLACK = [0,0,0]
        cropImg = cv2.copyMakeBorder(cropImg, lackOfyu, lackOfyd, lackOfxl, lackOfxr, cv2.BORDER_CONSTANT, value=BLACK)

        cropImg = cv2.resize(cropImg, (224, 224), interpolation=cv2.INTER_CUBIC)
        savePath = imagePath.replace("train","trainAligned")
        dirName = os.path.dirname(savePath)
        if not os.path.isdir(dirName):
            os.makedirs(dirName)
        cv2.imwrite(savePath, cropImg)
    except:
        # print(str(xl)+", "+str(xr)+", "+str(yu)+", "+str(yd))
        print(imagePath)

if __name__ == "__main__":
    p = Pool(processes = 20)
    p.map(crop, range(len(imagePaths)))
    p.close()