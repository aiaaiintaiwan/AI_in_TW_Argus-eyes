import os
import cv2
import config
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import roc_curve, auc, confusion_matrix
from eval_kit.detector import CelebASpoofDetector
from eval_kit.client import read_image

def model_eval(actual, pred):
    actual = list(map(lambda el:[el], actual)) 
    pred = list(map(lambda el:[el], pred)) 
    cm = confusion_matrix(actual, pred)
    print(cm)
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    accuracy = ((TP+TN))/(TP+FN+FP+TN)
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
    f_measure = (2*recall*precision)/(recall+precision)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)		
    error_rate = 1 - accuracy
    apcer = FP/(TN+FP)
    bpcer = FN/(FN+TP)
    acer = (apcer+bpcer)/2
    print('accuracy:  '+str(accuracy))
    print('precision:  '+str(precision))
    print('recall:  '+str(recall))
    print('f_measure:  '+str(f_measure))
    print('sensitivity:  '+str(sensitivity))
    print('specificity:  '+str(specificity))
    print('error_rate:  '+str(error_rate))
    print('apcer:  '+str(apcer))
    print('bpcer:  '+str(bpcer))
    print('acer:  '+str(acer))

def load_test_image():
    n=0
    images = []
    labels = []
    datas = [x.strip() for x in open(config.TEST_TXT_PATH)][:config.TEST_NUM]
    lenOfDatas=len(datas)
    for idx, data in enumerate(datas):
        try:
            n += 1
            imgPath, label = data.split(' ')
            imgPath = config.TEST_IMG_PATH + imgPath[10:]
            img = read_image(imgPath)
            images.append(img)
            labels.append(label)
        except:
            logging.info("Failed to read image: {}".format(os.path.join(LOCAL_IMAGE_PREFIX, image_id)))
            raise

        if n == config.TEST_BS or idx == lenOfDatas-1:
            n = 0
            tmpImages = images
            tmpLabels = labels
            images = []
            labels = []
            yield (tmpImages, tmpLabels)

def run_test(detector_class, imageIter):
    preds = []
    acts = []
    detector = detector_class()
    datas = [x.strip() for x in open(config.TEST_TXT_PATH)][:config.TEST_NUM]
    with open(config.TEST_ERR_IMG,"w") as f:
        for idx, (imgs, labels) in enumerate(imageIter):
            # try:
            probs = detector.predict(imgs)
            for i in range(len(probs)):
                '''
                groundT = int(labels[i])
                pred = 1 - np.argmax(probs[i])
                acts.append(groundT)
                preds.append(pred)
                '''
                
                groundT = int(labels[i])
                pred = round(probs[i][1].item())
                acts.append(groundT)
                preds.append(pred)
                
                if groundT!=pred:
                    index = idx*config.TEST_BS+i
                    imgPath = config.TEST_IMG_PATH + datas[index].split(' ')[0]
                    if groundT == 1:
                        f.write('Idex:%06d  ,GT:fake,  PD:real,  Path:%s\n'%(index, imgPath))
                    else:
                        f.write('Idex:%06d  ,GT:real,  PD:fake,  Path:%s\n'%(index, imgPath))
                    f.flush()
            '''            
            probs = detector.predict(imgs)
            for pb in probs:
                preds.append(round(pb[1]))
            acts = acts + [int(lb) for lb in labels]
            '''
            '''
            fpr, tpr, _ = roc_curve(acts,preds)
            roc_auc = auc(fpr, tpr)
            fig = plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            #fig.savefig('/tmp/roc.png')
            plt.show()
            '''
        # except:
        #     logging.info("faild predict: {}".format(idx))
    fpr, tpr, _ = roc_curve(acts,preds)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #fig.savefig('/tmp/roc.png')
    plt.show()
    model_eval(acts,preds)

if __name__ == '__main__':
    imageIter = load_test_image()
    run_test(CelebASpoofDetector, imageIter)
    