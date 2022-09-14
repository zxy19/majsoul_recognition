#cv from 1
# # 数据集
from ast import Num
import gc
import math
from operator import truediv
from platform import java_ver
import numpy as np
import torch
import torch.nn.functional as F #激励函数类（？大概是）
import torch.utils.data as Data #Dataloader所在组件类
import cv2 as cv;
import os
from math import *
import random
use_gpu = torch.cuda.is_available()
typesDic = [
#索子牌
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
#筒子牌
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
#万字牌
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
#   东   南    西   北   白   发   中
    "1z","2z","3z","4z","5z","6z","7z"
]
replaceDic = {}
pathD = "."
pathP = ".."
if ('__file__' in locals() or '__file__' in globals()):
    pathD=os.path.dirname(os.path.abspath(__file__))
    pathP=os.path.dirname(os.path.abspath(__file__))+"/.."
typesPic={}
for imgp in typesDic:
    tmImg=cv.imread(pathP+"/mahjong/"+imgp+".png")
    tmImg = cv.resize(tmImg,(60,40))
    typesPic[imgp] = tmImg
tmImg=cv.imread(pathP+"/mahjong/s5s.png")
tmImg = cv.resize(tmImg,(60,40))
replaceDic["5s"] = tmImg
tmImg=cv.imread(pathP+"/mahjong/s5m.png")
tmImg = cv.resize(tmImg,(60,40))
replaceDic["5m"] = tmImg
tmImg=cv.imread(pathP+"/mahjong/s5p.png")
tmImg = cv.resize(tmImg,(60,40))
replaceDic["5p"] = tmImg


def getDataReal(path):
    realIm = []
    realLb = []
    i = 1
    while os.path.exists(path+str(i)+".bmp") and os.path.exists(path+str(i)+".txt"):
        tmImg=cv.imread(path+str(i)+".bmp")
        tmImg = tmImg[5:75,5:75]
        tmImg = cv.resize(tmImg,(80,80))
        tmImg = pickShapeRGB(tmImg)
        lbfi = open(path+str(i)+".txt")

        i = i + 1

        lbtx = lbfi.readline()
        if lbtx == "":continue
        lbid = typesDic.index(lbtx)
        realIm.append(tmImg)
        realLb.append(lbid)
        
    return realIm,realLb
    
genCnt = 0
if not os.path.exists(pathD+"/saved/"):
    os.makedirs(pathD+"/saved/")
if not os.path.exists(pathD+"/results/"):
    os.makedirs(pathD+"/results/")
for root, dirs, files in os.walk(pathD+"/saved/", topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
for root, dirs, files in os.walk(pathD+"/results/", topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))




def maskLayerTB(img,botV,topV):
    top = np.uint8(np.where(img>topV,255,0))
    bott = np.uint8(np.where(img<botV,255,0))
    return cv.add(top,cv.subtract(img,bott))
def pickShapeRGB(img):
    imnew = img
    imnew = cv.blur(imnew,(5,5))
    return np.array(cv.split(imnew),float)/255
def generate(num):
    ret = []
    labels=[]
    curLabel = 0
    genCnt = 0
    for i in range(num):
        # if i % 5 == 0:
        #     curLabel = random.randint(0,len(realDataInput)-1)
        #     labels.append(realDataLabel[curLabel])
        #     im_apRes = np.array(realDataInput[curLabel],float)
        #     ret.append(im_apRes)
        #     continue
        curLabel = random.randint(0,len(typesDic)-1)
        labels.append(curLabel)
        
        img = typesPic[typesDic[curLabel]]
        if typesDic[curLabel]=="5s" or typesDic[curLabel]=="5p" or typesDic[curLabel]=="5m":
            if random.random()>0.5:
                img = replaceDic[typesDic[curLabel]]
        
        if random.random()>0.5:
            img = cv.resize(img.copy(),(random.randint(30,60),random.randint(30,60)))
        if random.random()>0.9:#牌面变红
            imshape = img.shape[:2]
            filter1 = cv.merge(np.array([
                np.ones(imshape,dtype=np.uint8)*0,
                np.ones(imshape,dtype=np.uint8)*0,
                np.ones(imshape,dtype=np.uint8)*30
            ]))
            filter2 = cv.merge(np.array([
                np.ones(imshape,dtype=np.uint8)*30,
                np.ones(imshape,dtype=np.uint8)*30,
                np.ones(imshape,dtype=np.uint8)*0
            ]))
            img = cv.add(img,filter1)
            img = cv.subtract(img,filter2)


        bri = random.randint(0,7)
        if bri>=5:#全局降亮
            bs = 0.1*(bri - 5) + 0.8
            img = np.uint8(np.float32(img)*bs)
        elif random.random()>0.4:#全局提亮
            imshape = img.shape[:2]
            bint = random.randint(20,50)
            filt = cv.merge(np.array([
                np.ones(imshape,dtype=np.uint8)*bint,
                np.ones(imshape,dtype=np.uint8)*bint,
                np.ones(imshape,dtype=np.uint8)*bint
            ]))
            img = cv.add(img,filt)
            
        if random.random()>0.6:#全局渐变提亮
            imshape = img.shape
            imFilter = np.zeros(imshape,dtype=np.uint8)
            center = random.randint(1,imshape[1]-1)
            for i in range(center,max(center-20,0),-1):
                imFilter[:,i,:]=abs(20-center+i)*4
            for i in range(center+1,min(center+20,imshape[1]-1),1):
                imFilter[:,i,:]=abs(20-i+center)*4
            rows,cols = imshape[:2]
            M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
            imFilter = cv.warpAffine(imFilter,M,(cols,rows))
            # cv.imshow("lightMask",imFilter)
            img = cv.add(img,imFilter)


        if random.random()>0.6:#随机局部提亮
            imshape = img.shape[:2]
            height,width = imshape
            filter_array = np.array([
                np.ones(imshape,dtype=np.uint8)*0,
                np.ones(imshape,dtype=np.uint8)*0,
                np.ones(imshape,dtype=np.uint8)*0
            ])
            l = random.randint(0,int(width/2))
            r = random.randint(l,width-1)
            t = random.randint(0,int(height/2))
            b = random.randint(t,height-1)
            filter_array[:,l:r,t:b]=random.randint(10,30)
            filter = cv.merge(filter_array)
            img = cv.add(img,filter)
        
        if random.random()>0.6:
            imshape = img.shape
            lightMask = np.zeros(imshape,np.uint8)
            if random.random()>0.5:
                x1 = random.randint(0,60)
                y1 = 0
                x2 = 0
                y2 = int(x1/3*2)
                ptn1 = (x1,y1)
                ptn2 = (x2,y2)
            else:
                x1 = random.randint(0,60)
                y1 = 0
                x2 = 60
                y2 = int(40 - (60 - x1)/3*2)
                ptn2 = (x1,y1)
                ptn1 = (x2,y2)
            cv.line(lightMask,ptn1,ptn2,(160,160,160),10)
            img = cv.add(img,lightMask)
        # 边框处理
        if random.randint(0,10) > 3:#扩充边缘
            height,width = img.shape[:2]
            bbc=(random.randint(0,255), random.randint(0,255), random.randint(0,255))  
            #bbc=(200,200,200)  
            bordWid1 = random.randint(0,2)
            bordWid2 = bordWid1
            bordWid3 = bordWid1
            bordWid4 = bordWid1
            if random.randint(0,10) > 5:bordWid1 = random.randint(5,20)
            if random.randint(0,10) > 5:bordWid2 = random.randint(5,20)
            if random.randint(0,10) > 5:bordWid3 = random.randint(5,20)
            if random.randint(0,10) > 5:bordWid4 = random.randint(5,20)
            img = cv.copyMakeBorder(img,bordWid1,bordWid2,bordWid3,bordWid4,cv.BORDER_CONSTANT,value=bbc)
        elif random.randint(0,10) > 4:#裁剪边缘
            height,width = img.shape[:2]
            bordWid1 = random.randint(1,5)
            bordWid2 = random.randint(1,5)
            bordWid3 = random.randint(1,5)
            bordWid4 = random.randint(1,5)
            img = img[bordWid3:height-bordWid4,bordWid1:width-bordWid2]

        bc=(random.randint(0,255), random.randint(0,255), random.randint(0,255))  

        #透视变换
        if random.random()>0.3:
            height,width = img.shape[:2]
            heightNew,widthNew = img.shape[:2]
            #   1---3
            #   |   |
            #   2---4
            x1=0
            y1=0
            x2=0
            y2=heightNew-1
            x3=widthNew-1
            y3=0
            x4=widthNew-1
            y4=heightNew-1
            didTime = 0
            while random.random()>0.3:
                if didTime>3:break
                else: didTime = didTime+1
                op = random.randint(0,3)
                dlt = random.randint(-5,20)
                dlt_dlt = random.randint(-6,6)
                if op == 0:
                    x1 = x1 + dlt
                    x3 = x3 + dlt
                elif op == 1:
                    x2 = x2 + dlt
                    x4 = x4 + dlt
                elif op == 2:
                    y1 = y1 + dlt
                    y2 = y2 + dlt
                elif op == 3:
                    y3 = y3 + dlt
                    y4 = y4 + dlt
        
            if x1 < 0 or x2 < 0:
                dx = max(0 - x1,0-x2)
                x1 = x1 + dx
                x2 = x2 + dx
                x3 = x3 + dx
                x4 = x4 + dx
            if y1 < 0 or y2 < 0:
                dy = max(0 - y1,0-y2)
                y1 = y1 + dy
                y2 = y2 + dy
                y3 = y3 + dy
                y4 = y4 + dy
            widthNew = max(x3,x4)
            heightNew = max(y4,y2)
            

            ptn1 = np.array(
                [
                    (x1,y1),
                    (x3,y3),
                    (x4,y4),
                    (x2,y2)
                ]
            ).astype('float32')
            ptn2 = np.array(
                [[0,0],[width,0],[width,height],[0,height]]
            ).astype('float32')
            Tm = cv.getPerspectiveTransform(ptn2,ptn1)
            img = cv.warpPerspective(img,Tm,dsize=(widthNew,heightNew),borderValue=bc)
        
        if random.random()>0.5:#旋转变换
            x= random.randint(0, 4)
            degree=x * 90 + random.randint(0,16) - 8
            height,width = img.shape[:2]

            M = cv.getRotationMatrix2D((width/2,height/2),degree,1)
            heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
            widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

            M[0, 2] += (widthNew - width) / 2  
            M[1, 2] += (heightNew - height) / 2
            bc=(random.randint(0,255), random.randint(0,255), random.randint(0,255))  
            img = cv.warpAffine(img,M,(widthNew,heightNew), 
                borderValue=bc)
        im_otest = img
        img = cv.resize(img,(80,80))

        
        im_genRes = img
        im_genRes = pickShapeRGB(im_genRes)

        #DEBUG
        # print("[GENERATOR]->"+typesDic[curLabel])
        # cv.imshow("oim2",img)
        # cv.imshow("oim",cv.merge(np.uint8(im_genRes * 255)))
        # cv.waitKey(0)
        #DEBUG
        
        #DEBUG:输出图片
        # cv.imwrite(pathD+"/../gen/"+str(genCnt)+".bmp",im_genRes)
        # genCnt = genCnt+1

        #ret.append([im_apRes])
        ret.append(im_genRes)
    return torch.FloatTensor(np.array(ret)),torch.tensor(labels)

class ResidualBlock(torch.nn.Module):
    def __init__(self,channels_in,channels_mid,channels_out,kernerSize=3):
        super(ResidualBlock,self).__init__()
        padding_noSizeChange = int(kernerSize/2)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels_in,channels_mid,kernerSize,1,padding_noSizeChange),
            torch.nn.BatchNorm2d(channels_mid),
            torch.nn.LeakyReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels_mid,channels_out,kernerSize,1,padding_noSizeChange),
            torch.nn.BatchNorm2d(channels_out)
        )
        self.conv_short =torch.nn.Conv2d(channels_in,channels_out,1,1,0)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.conv_short(x)
        return out

#采用卷据层
class Net(torch.nn.Module):
    def __makeLayerBlock(self,block,batchNormBlock,outputChannel,activation,*arg):
        tmpRet = torch.nn.Sequential()
        tmpRet.append(block(*arg))
        tmpRet.append(batchNormBlock(outputChannel))
        tmpRet.append(activation())
        return tmpRet
    def __init__(self):
        super(Net, self).__init__()
        self.layer2d = torch.nn.ModuleList()
        self.layer1d = torch.nn.ModuleList()
        #80 * 80 * 3
        self.layer2d.append(self.__makeLayerBlock(ResidualBlock,torch.nn.BatchNorm2d,12,torch.nn.LeakyReLU,3,6,12))
        #80 * 80 * 12
        self.layer2d.append(torch.nn.MaxPool2d(2))
        #40 * 40 * 36
        self.layer2d.append(self.__makeLayerBlock(ResidualBlock,torch.nn.BatchNorm2d,60,torch.nn.LeakyReLU,12,48,60))
        #40 * 40 * 60
        self.layer2d.append(torch.nn.MaxPool2d(2))
        #20 * 20 * 84
        self.layer2d.append(self.__makeLayerBlock(ResidualBlock,torch.nn.BatchNorm2d,108,torch.nn.LeakyReLU,60,96,108))
        #20 * 20 * 108
        self.layer2d.append(torch.nn.MaxPool2d(2))
        #10 * 10 * 132
        self.layer2d.append(self.__makeLayerBlock(ResidualBlock,torch.nn.BatchNorm2d,156,torch.nn.LeakyReLU,108,144,156))
        #10 * 10 * 156
        self.layer2d.append(torch.nn.MaxPool2d(2))
        #5 * 5 * 180
        self.layer2d.append(self.__makeLayerBlock(ResidualBlock,torch.nn.BatchNorm2d,204,torch.nn.LeakyReLU,156,192,204))
        #5 * 5 * 228
        self.layer1d.append(self.__makeLayerBlock(torch.nn.Linear,torch.nn.BatchNorm1d,1000,torch.nn.LeakyReLU,5*5*204,1000))
        self.layer1d.append(self.__makeLayerBlock(torch.nn.Linear,torch.nn.BatchNorm1d,1000,torch.nn.LeakyReLU,1000,1000))
        self.layer1d.append(torch.nn.Linear(1000,34))

    def forward(self, x):
        out = x
        for layer in self.layer2d:
            out = layer(out)
            out = torch.nn.functional.dropout2d(out,0.00005)
        out = out.view(out.size(0),-1)
        for layer in self.layer1d:
            out = layer(out)
            out = torch.nn.functional.dropout(out,0.00005)
        return out
def isPredictCorrect(pre,dat):
    return torch.argmax(pre) == dat

net = Net()
if os.path.exists(pathD+"/weight.pth"):
    net.load_state_dict(torch.load(pathD+"/weight.pth"))
    print("PRETRAIN WEIGHTS LOADED")

# #读取一些真实标注的数据，用于更好的训练和跟踪效果
# realDataInput = []
# realDataLabel = []
# _im,_lb = getDataReal(pathD+"/real1/")
# realDataInput.extend(_im)
# realDataLabel.extend(_lb)
# _im,_lb = getDataReal(pathD+"/real2/")
# realDataInput.extend(_im)
# realDataLabel.extend(_lb)
# _im,_lb = getDataReal(pathD+"/real3/")
# realDataInput.extend(_im)
# realDataLabel.extend(_lb)
# _im,_lb = getDataReal(pathD+"/real4/")
# realDataInput.extend(_im)
# realDataLabel.extend(_lb)
# _im,_lb = getDataReal(pathD+"/real5/")
# realDataInput.extend(_im)
# realDataLabel.extend(_lb)
# _im,_lb = getDataReal(pathD+"/real6/")
# realDataInputVolTensor=torch.FloatTensor(np.array(_im,float))
# realDataLabelVol = _lb
# if(use_gpu):
#     realDataInputVolTensor=realDataInputVolTensor.cuda()


optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 传入 net 的所有参数, 学习率
lrschedu = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.1,verbose=True)
loss_func = torch.nn.CrossEntropyLoss()      #交叉熵损失函数
if(use_gpu):
    net.cuda()
    loss_func.cuda()
    print("USING CUDA ACCELRATING")
#TRAIN SET
meanLoss = []
correctRateList = []
for epoch in range(3000):
    net.train()
    StepMinLoss = 1E10
    for step in range(30):
        data_x,data_y = generate(64)
        print("EPOCH %d,STEP %d Generated"%(epoch,step),end="\r")
        if(use_gpu):
            data_x=data_x.cuda()
            data_y=data_y.cuda()
        prediction = net(data_x)     # 喂给 net 训练数据 x, 输出预测值
        loss = loss_func(prediction, data_y)     # 计算两者的误差
        if(use_gpu):
            loss = loss.cpu()
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()
        print("EPOCH %d,\tSTEP %d---\tLOSS:%f"%(epoch,step,loss.mean()))
        StepMinLoss = min(StepMinLoss,loss.mean())
    lrschedu.step(StepMinLoss)
    net.eval()
    currectCnt=0
    wrongCnt=0
    #验证集运算
    data_x,data_y = generate(64)
    data_oy = data_y.numpy()
    data_ox = np.uint8(data_x * 255)
    if(use_gpu):
        data_x=data_x.cuda()
        data_y=data_y.cuda()
    prediction = net(data_x)
    
    loss = loss_func(prediction, data_y)
    if(use_gpu):
            loss = loss.cpu()
            prediction = prediction.cpu()
            data_y=data_y.cpu()
    wrongStr = "" 
    resImg = np.zeros((80*8,80*16,3),np.uint8)

    for i in range(len(prediction)):
        if i < 64:
            inputIm = cv.merge(data_ox[i])
            ansIm = cv.resize(typesPic[typesDic[torch.argmax(prediction[i])]],(70,70))
            resImg[(i%8)*80:(i%8)*80+80,int(i/8)*160:int(i/8)*160+80,:]=inputIm
        if isPredictCorrect(prediction[i],data_y[i]):
            if i < 64:ansIm = cv.copyMakeBorder(ansIm,5,5,5,5,cv.BORDER_CONSTANT,value=(0,255,0))
            currectCnt = currectCnt + 1
        else:
            if i < 64:ansIm = cv.copyMakeBorder(ansIm,5,5,5,5,cv.BORDER_CONSTANT,value=(0,0,255))
            wrongCnt = wrongCnt + 1
            if wrongCnt < 6:
                wrongStr = wrongStr + typesDic[data_y[i]] + "->" + typesDic[torch.argmax(prediction[i])] + ";\n"
            elif wrongCnt == 6:
                wrongStr = wrongStr + "......"
        if i < 64:resImg[(i%8)*80:(i%8)*80+80,int(i/8)*160+80:int(i/8)*160+160,:]=ansIm
    cv.imwrite(pathD+"/results/BatchVolRes_"+str(epoch)+".bmp",resImg)
    correctRate = (float(currectCnt)/float(currectCnt+wrongCnt)*100.0)
    print("-->EPOCH [%d] Finished,\tVOL Hit:[%f%%]\n-->\tVOL MEAN LOSS:[%f]\tBATCH MIN LOSS:[%f]"%(epoch,correctRate,loss.mean(),StepMinLoss))
    print("----->WRONG CASES:\n"+wrongStr)
    # #真实数据集校验
    # currectCnt=0
    # wrongCnt=0
    # prediction = net(realDataInputVolTensor)
    # wrongStr = "" 
    # for i in range(len(prediction)):
    #     if isPredictCorrect(prediction[i],realDataLabelVol[i]):
    #         currectCnt = currectCnt + 1
    #     else:
    #         wrongCnt = wrongCnt + 1
    #         if wrongCnt < 10:
    #             wrongStr = wrongStr + typesDic[realDataLabelVol[i]] + ";\n"
    #         elif wrongStr == 10:
    #             wrongStr = wrongStr + "......"
    # correctRate = (float(currectCnt)/float(currectCnt+wrongCnt)*100.0)
    # print("----->EPOCH %d VERIFY,\tREAL DATA Hit:%f%%\n"%(epoch,correctRate))
    # print("----->WRONG CASES:\n"+wrongStr)
    if (epoch+1) % 5 == 0:
        torch.save(net.state_dict(),pathD+"/saved/model_weight_echop_%d.pth"%(epoch+1))
        print("MODEL WEIGHT SAVED ==> model_weight_echop_%d.pth")
    print("==================================\n\n")