import json
import os
import cv2 as cv
import torch
import torch.nn.functional as F #激励函数类（？大概是）
import numpy as np

import asyncio
from threading import Timer
import win32gui, win32ui, win32con, win32api,win32print
import websockets

ENABLE_CUDA = True
REC_MAXBATCHSIZE = 1000
use_gpu = torch.cuda.is_available() and ENABLE_CUDA

pathD = "."
pathP = ".."
if ('__file__' in locals() or '__file__' in globals()):
    pathD=os.path.dirname(os.path.abspath(__file__))
    pathP=os.path.dirname(os.path.abspath(__file__))+"/.."
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

def pickShapeRGB_DRK(img):
    r,g,b = cv.split(img)
    gmask = np.uint8(np.where(r<150,0,255))
    gmask = cv.bitwise_and(gmask,np.uint8(np.where(g<150,0,255)))
    gmask = cv.bitwise_and(gmask,np.uint8(np.where(b<150,0,255)))
    
    rmid = min(int((r[30:50,30:50].max()+10)),200)
    r = cv.bitwise_and(np.uint8(np.where(r<rmid,0,255)),gmask)
    gmid = min(int((g[30:50,30:50].max()+10)),200)
    g = cv.bitwise_and(np.uint8(np.where(g<gmid,0,255)),gmask)
    bmid = min(int((b[30:50,30:50].max()+10)),200)
    b = cv.bitwise_and(np.uint8(np.where(b<bmid,0,255)),gmask)
    
    imnew = cv.merge((np.uint8(r),np.uint8(g),np.uint8(b)))
    # cv.imshow("o",img)
    # cv.imshow("merge",imnew)
    
    # cv.imshow("r",np.uint8(r))
    # cv.imshow("g",np.uint8(g))
    # cv.imshow("b",np.uint8(b))
    # cv.waitKey(0)
    return np.array(cv.split(imnew),np.uint8)
def maskLayerTB(img,botV,topV):
    top = np.uint8(np.where(img>topV,255,0))
    bott = np.uint8(np.where(img<botV,255,0))
    return cv.add(top,cv.subtract(img,bott))
def pickShapeRGB(img):
    imnew = img
    imnew = cv.blur(imnew,(5,5))
    return np.array(cv.split(imnew),float)/255
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
            torch.nn.BatchNorm2d(channels_out),
        )
        self.conv_short =torch.nn.Conv2d(channels_in,channels_out,1,1,0)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.conv_short(x)
        return out

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
            #out = torch.nn.functional.dropout2d(out,0.00005)
        out = out.view(out.size(0),-1)
        for layer in self.layer1d:
            out = layer(out)
            #out = torch.nn.functional.dropout(out,0.00005)
        return out
model = Net()
if(use_gpu):
    model.cuda()
    map_location=torch.device('cuda')
else:
    map_location=torch.device('cpu')
model.load_state_dict(torch.load(pathD+"/weight.pth",map_location))
model.eval()
#截图
def window_capture():
    # return cv.imread("sence2.png")
    # classname = "UnityWndClass"
    # titlename = "雀魂麻將"
    # #获取句柄
    # hwnd = win32gui.FindWindow(classname, titlename)
    # 
    hwnd = 0 
    #hwnd = 0  # 窗口的编号，0号表示当前活跃窗口
    # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    hwndDC = win32gui.GetWindowDC(hwnd)
    # 根据窗口的DC获取mfcDC
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    # mfcDC创建可兼容的DC
    saveDC = mfcDC.CreateCompatibleDC()
    # 创建bigmap准备保存图片
    saveBitMap = win32ui.CreateBitmap()
    # 获取监控器信息
    hDC = win32gui.GetDC(0)
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    # print w,h　　　#图片大小
    # 为bitmap开辟空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    # 高度saveDC，将截图保存到saveBitmap中
    saveDC.SelectObject(saveBitMap)
    # 截取从左上角（0，0）长宽为（w，h）的图片
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
    signedIntsArray = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype = 'uint8')
    img = img.reshape(( h ,w, 4))
    img = cv.cvtColor(img,cv.COLOR_RGBA2RGB)
    # cv.imshow("i",img)
    # cv.waitKey(0)
    return img
#四点取中心
def getCenter(ct):
    return np.array([(ct[0][0]+ct[1][0]+ct[2][0]+ct[3][0])/4,(ct[0][1]+ct[1][1]+ct[2][1]+ct[3][1])/4])
#轮廓取角点
def getFourPoints(contours):
    retPtn=[contours[0],contours[0],contours[0],contours[0]]
    for ptn in contours:
        xmCnt=0     #
        ymCnt=0
        for aptn in contours:
            if aptn[0]<ptn[0]:
                xmCnt=xmCnt+1
            if aptn[1]<ptn[1]:
                ymCnt=ymCnt+1
        if xmCnt<2 and ymCnt<2:
            retPtn[0]=ptn       #LT
        if xmCnt>=2 and ymCnt<2:
            retPtn[1]=ptn       #RT
        if xmCnt>=2 and ymCnt>=2:
            retPtn[2]=ptn       #RB
        if xmCnt<2 and ymCnt>=2:
            retPtn[3]=ptn       #LB
    
    return np.array(retPtn)
#轮廓过滤
def dealContour(orgContour,img):
    shapeContour = []
    for contour in orgContour:
        hull = cv.convexHull(contour)
        approx = cv.approxPolyDP(hull,10,True)
        rectt = cv.minAreaRect(approx)
        x,y,w,h = cv.boundingRect(contour)
        rectc = rectt
        rect = cv.boxPoints(rectc)
        recto = cv.boxPoints(rectt)
        rect = np.int0(rect)
        area_c = cv.contourArea(approx)
        area_r = cv.contourArea(rect)
        area_or = cv.contourArea(recto)
        # cv.imshow("test",img[y:y+h,x:x+w,:])
        # cv.waitKey(0)
        if(
            min(rectt[1][1],rectt[1][0])>10
            and abs(rectt[1][0]-rectt[1][1])/min(rectt[1][1],rectt[1][0]) < 3 
            and (abs(rectt[1][0]-rectt[1][1])/min(rectt[1][1],rectt[1][0])>0.3 or area_c<6000)
            and area_c < 30000 
            and area_c > 1000
            and (area_or-area_c)/area_or<0.25
            ):
            approx[:,:,0]=approx[:,:,0] - x
            approx[:,:,1]=approx[:,:,1] - y
            shapeContour.append((x,y,w,h,approx))
            # cv.imshow("test",img[y:y+h,x:x+w,:])
            # cv.waitKey(0)
            pass
        elif area_c > 500:
            # cv.imshow("test",img[y:y+h,x:x+w,:])
            # cv.waitKey(0)
            pass
    return shapeContour
#遮罩图片
def makeMask(rgb,rTh,gTh,bTh,lvl=20):
    mask1 = cv.inRange(rgb, np.array([0, 0, rTh - lvl]), np.array([255,255,rTh]))
    mask2 = cv.inRange(rgb, np.array([0, gTh - lvl, 0]), np.array([255,gTh,255]))
    mask3 = cv.inRange(rgb, np.array([bTh - lvl, 0, 0]), np.array([bTh,255,255]))
    mask = cv.bitwise_and(mask1,cv.bitwise_and(mask2,mask3))
    return mask
typeMask = cv.imread(pathD+"/mask.png")
typeMask = cv.resize(typeMask,(1920,1080))
def getType(centerPtn):
    ptnValGrp = typeMask[centerPtn[1],centerPtn[0],:]
    ptnVal = int((np.array((1,256,65525),np.int64)*np.array(ptnValGrp,np.int64)).sum())
    return ptnVal
#图片切分
def cutPic(img):
    oheight,owidth = img.shape[:2]
    yScal = 1.0*oheight/1080.0
    xScal = 1.0*owidth/1920.0
    img=cv.resize(img,(1920,1080))
    maskV=(
        (255,255,255,50),
        (200,200,200,30),
        (240,200,200,50),
        (170,170,170,30)
    )
    output = []
    ptns = []
    oPtnsss = []
    i=0
    for thres in maskV:
        mask = makeMask(img,*thres)
        # cv.imshow("mask",mask)
        # cv.waitKey(0)
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours=dealContour(contours,img)
        # im_tst = np.uint8(img*1)
        # im_tst=cv.drawContours(im_tst,contours,-1,(0,255,0))
        # cv.imwrite("./page/res_debug.png",im_tst)
        # cv.imshow("oim",im_tst)
        # cv.waitKey(0)
        
        for x,y,w,h,shapeCnt in contours:
            i=i+1
            shape = (h,w,3)
            mask = np.zeros(shape,np.uint8)
            mask = cv.bitwise_not(cv.drawContours(mask,[shapeCnt],-1,(255,255,255),-1))
            dst = img[y:y+h,x:x+w,:]
            dst = cv.bitwise_or(dst,mask)
            dst = cv.resize(dst,(80,80))
            # cv.imshow("mask",mask)
            # cv.imshow("dst",dst)
            # cv.waitKey(0)
            # res = cv.split(dst)
            # print(i)
            res = pickShapeRGB(dst)
            #res = np.array([dst])
            # cv.imwrite("cutten\\"+str(i-1)+".bmp",cv.merge(np.uint8(res*255)))
            # cv.imwrite("cutten\\"+str(i-1)+"_MASK.bmp",dst_ctr)
            output.append(res)
            centerPtno = (int(x+w/2),int(y+h/2))
            oPtnsss.append(centerPtno)
            centerPtn=(int(centerPtno[0]*xScal),int(centerPtno[1]*yScal))
            ptns.append(centerPtn)
    return output,ptns,oPtnsss
#识别结果处理
def dealRes(modelOpt):
    res = []
    modelOpt = modelOpt.detach().numpy()
    for i in modelOpt:
        ans = i.argmax()
        res.append(typesDic[ans])
    return res
#识别
def rec(img):
    print("切分")
    opt,ptn,optn=cutPic(img)
    im_tst = img
    print("识别")
    iptSize = len(opt)
    if iptSize == 0:
        return []
    res = []
    input = np.array(opt,float)
    with torch.no_grad():
        for i in range(0,iptSize,REC_MAXBATCHSIZE):
            iptTensor = torch.FloatTensor(input[i:min(iptSize,i+REC_MAXBATCHSIZE)])
            if(use_gpu):
                iptTensor = iptTensor.cuda()
            tres = model(iptTensor)
            if(use_gpu):
                tres = tres.cpu()
            tres = dealRes(tres)
            res.extend(tres)
    ret = []
    for i in range(len(ptn)):
        ptnType = getType(optn[i])
        if ptnType == 0:
            continue
        ret.append({
            "type":ptnType,
            "center":{
                "x":ptn[i][1],
                "y":ptn[i][0]
            },
            "res":res[i]
        })
        # with open("cutten\\"+str(i)+".txt","w+") as f:
        #     f.write(res[i])
        im_tst=cv.rectangle(im_tst, np.int32(ptn[i])+np.array([-1,25]), np.int32(ptn[i])+np.array([65,-25],np.int32), (0,0,0), -1)
        im_tst = cv.putText(im_tst,str(i),np.int32(ptn[i]),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=0.8,color=(0,255,0),thickness=2)
        im_tst = cv.putText(im_tst,res[i],np.int32(ptn[i])+np.array([0,25]),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=0.8,color=(0,0,255),thickness=2)
    cv.imwrite("./page/res.png",im_tst)
    return ret
def recScr(server):
    print("正在处理图片")
    img = window_capture()
    
    res = rec(img)
    print("更新数据")
    websockets.broadcast(server.websockets,json.dumps({"act":"list","res":res}))
async def echo(websocket):
    try:
        while True:
            message = await websocket.recv()
    finally:
        pass
class RepeatingTimer(Timer): 
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)
async def main():
    async with websockets.serve(echo, "localhost", 19902) as server:
        t = RepeatingTimer(interval= 1.0, function=recScr,kwargs={"server":server})
        t.start()
        await asyncio.Future()
        print("closed")

asyncio.run(main())
