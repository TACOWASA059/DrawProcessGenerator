import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
import glob
from PIL import Image
import re

class DrawProcessGenerator:
    def __init__(self, filename,output_directory,threshold=10):
        self.threshold=threshold
        self.img = cv2.imread(filename)
        self.directory = output_directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        print("directory:"+self.directory)
        self.pre_processing()
        self.contour_detection()
        self.plot_draft()
        self.plot_color()
        self.output_gif()

        self.ax.cla
        plt.clf()
        plt.cla()
        plt.close()
    def pre_processing(self):
        #morphology勾配
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # edge detection using Morphological Gradient
        self.dst = cv2.morphologyEx(self.img, cv2.MORPH_GRADIENT, kernel, iterations=1)
        kernel = np.ones((3,3), np.uint8)
        self.dst = cv2.dilate(self.dst,kernel,iterations = 1)
        # erase double contour
        self.dst = cv2.morphologyEx(self.dst, cv2.MORPH_CLOSE, np.ones((3, 3)))
        self.dst=np.array(self.dst,dtype='uint8')#rgbになっているはず
        #明度の大きさを元に二値化する
        self.dst = cv2.cvtColor(self.dst,cv2.COLOR_RGB2HSV) 
        self.dst[:,:,2] = np.where(self.dst[:,:,2] > self.threshold, 255, 0)
        self.dst = cv2.cvtColor(self.dst,cv2.COLOR_HSV2RGB) 
        cv2.imwrite("input1.png",self.dst)
        return self.dst
    def contour_detection(self):
        gray = cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
        #gray = cv2.Canny(gray, 50, 150)
        # apply a threshold to create a binary image
        ret, thresh =  cv2.threshold(gray,64,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # find the contours in the binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,)
        # remove inappropriate contours
        contours = list(filter(lambda x: cv2.contourArea(x) > len(self.img)*len(self.img[0])/4800, contours))
        self.contours = list(filter(lambda x: cv2.contourArea(x) < int(len(self.img)*len(self.img[0])/1.92), contours))
        return self.contours
    def plot_draft(self):
        self.fig, self.ax = plt.subplots()
        self.ax.axis("off")
        self.ax.set_xlim(0, self.img.shape[1])
        self.ax.set_ylim(self.img.shape[0], 0)
        self.ax.set_aspect('equal')
        self.count=0
        for i in range(len(self.contours)):
            cnt = self.contours[i]
            x = [p[0][0] for p in cnt]
            y = [p[0][1] for p in cnt]
            index=int(len(cnt)/50)
            for j in range(0, len(cnt), 50):
                self.ax.plot(x[j:j+50], y[j:j+50], lw=1, color="#000000")
                self.fig.savefig(self.directory+str(self.count)+".png",dpi=300,transparent=True)
                self.count+=1
            self.ax.plot(x[index:]+[x[0]], y[index:]+[y[0]], lw=1, color="#000000")
            self.fig.savefig(self.directory+str(self.count)+".png",dpi=300,transparent=True)
            self.count+=1
        print("draft_ok")
    def plot_color(self):
        self.img=cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)#変更が必要みたい
        for i in range(len(self.contours)):
            cnt = self.contours[i]
            #maskの大きさをimgと同じにする
            mask = np.zeros(self.img.shape, np.uint8)
            #輪郭をマスクに記述する
            mask=cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
    
            #maskをグレイスケールに変換
            #以下のbitwise_andではmaskはgrayスケールでないといけない
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)    
            #maskをかける(ただし返り値はRGB)
            img_masked = cv2.bitwise_and(self.img, self.img, mask=mask)
            draw_pixel_per_step=len(self.img)*len(self.img[0])/500
            #領域でnon-zeroのピクセル数
            mask_region_count=np.count_nonzero(mask)
            #領域のサイズに応じた分割数を指定する
            div_num=math.ceil(mask_region_count/draw_pixel_per_step)
            #マスク領域をdiv_num等分する
            step = int(mask_region_count / div_num)
            #255の部分を取得
            indices = np.where(mask == 255)
            for j in range(div_num):
                mask_temp = np.zeros(mask.shape, np.uint8)
                mask_temp[indices[0][0:(j+1)*step],indices[1][0:(j+1)*step]] = 255
                img_masked_temp = cv2.bitwise_and(self.img, self.img, mask=mask_temp)
                img_masked_temp = np.dstack((img_masked_temp, mask_temp))
        
                self.ax.imshow(img_masked_temp)
                self.fig.savefig(self.directory+str(self.count)+".png",dpi=300,transparent=True)
                self.count+=1
            if div_num*step<len(indices[0])-1:
                mask_temp = np.zeros(mask.shape, np.uint8)
                mask_temp[indices[0][div_num*step:],indices[1][div_num*step:]] = 255
                img_masked_temp = cv2.bitwise_and(self.img, self.img, mask=mask_temp)
                img_masked_temp = np.dstack((img_masked_temp, mask_temp))
        
                self.ax.imshow(img_masked_temp)
                self.fig.savefig(self.directory+str(self.count)+".png",dpi=300,transparent=True)
                self.count+=1
        print("color_ok")
    def output_gif(self):
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]

        #files = sorted(glob.glob('data/*.jpg'), key=natural_keys)
        path1 = self.directory
        print(path1)
        a1 = sorted(glob.glob( path1 + "*.png" ),key=natural_keys)
        img1 = list( map( lambda file0: Image.open( file0 ), a1 ) )

        file2 = path1 + 'video1.gif'
        img1[0].save( file2, save_all=True, append_images = img1[1:], duration=10, loop=0 )

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python DrawProcessGenerator.py <filename>")
        sys.exit()
    filename = sys.argv[1]
    drawprocess=DrawProcessGenerator(filename=filename,output_directory="move0/")