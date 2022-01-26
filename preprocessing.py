import cv2
import numpy as np
import xlsxwriter as xls
from skimage.feature import greycomatrix, greycoprops
import math
from scipy import stats

# Baris citra jenis mangga
dataset_excel = xls.Workbook('dataset.xlsx')
sheet         = dataset_excel.add_worksheet()
jenis_mangga  = ['Aafush','Dasheri','Jamadar','Kesar','Rajapuri','Totapuri']
jml_per_data  = 130

sheet.write(0,0,'Data Gambar')
kolom = 1

# Kolom fitur glcm
fitur_glcm = ['correlation','homogeneity', 'dissimilarity', 'contrast','energy','ASM']
angle = ['0','45','90','135']
for i in fitur_glcm:
    for j in angle:
        sheet.write(0,kolom,i+" "+j)
        kolom+=1

# Kolom fitur bentuk
fitur_bentuk = ['metric','eccentricity']
for i in fitur_bentuk:
    sheet.write(0,kolom,i)
    kolom+=1


# Kolom fitur hsv
fitur_hsv = ['hue','saturation','values']
for i in fitur_hsv:
    sheet.write(0,kolom,i)
    kolom+=1

sheet.write(0,kolom,'label')
kolom+1
baris=1

# Proses perulangan untuk masing masing data
for i in jenis_mangga:
    for j in range(1,jml_per_data+1):
        kolom=0
        file_name = "img_dataset/"+i+str(j)+".bmp"
        print(file_name)
        sheet.write(baris,kolom,file_name)
        kolom+=1

        # Preprocessing
        img         = cv2.imread(file_name,1)
        grayscale   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                     # Rubah dalam grayscale
        ret, img1   = cv2.threshold(grayscale,129,255,cv2.THRESH_BINARY_INV)    # Melakukan tresholding
        img1        = cv2.dilate(img1.copy(),None,iterations=5)                 # Menutup lubang pada citra
        img1        = cv2.erode(img1.copy(),None,iterations=5)                  # Mengikis pingiran citra hasil dari dilasi
        b,g,r       = cv2.split(img)
        rgba        = [b,g,r, img1]
        dst         = cv2.merge(rgba,4)

        # Cropping
        contours, hierarchy = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        selected            = max(contours,key=cv2.contourArea)         #Menentukan kontur terbesar dari citra
        x,y,w,h             = cv2.boundingRect(selected)
        png                 = dst[y:y+h,x:x+w]
        gray                = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)

        # Ekstraksi glcm
        distances = [5]
        angles   = [0,np.pi/4,np.pi/2,3*np.pi/4]
        levels   = 256
        symmetric=True
        normed   =True

        glcm = greycomatrix(gray, distances, angles, levels, symmetric, normed)
        glcm_props = [propery for name in fitur_glcm for propery in greycoprops(glcm, name)[0]]
        for item in glcm_props:
            sheet.write(baris,kolom,item)
            kolom+=1

        # Ekstraksi bentuk 
        # Eccentricity
        dimensions  = png.shape
        height      = png.shape[0]
        width       = png.shape[1]
        mayor       = max(height,width)
        minor       = min(height,width)
        eccentricity = math.sqrt(1-((minor*minor)/(mayor*mayor)))

        # Metric
        height1       = img.shape[0]
        width1        = img.shape[1]
        edge          = cv2.Canny(img,100,200)

        k=0
        keliling=1
        while k<height1:
            l=0
            while l<width1:
                if edge[k,1]==255:
                    keliling=keliling+1
                l=l+1
            k=k+1

        k=0
        luas=1
        while k<height1:
            l=0
            while l<width1:
                if img1[k,1]==255:
                    luas=luas+1
                l=l+1
            k=k+1

        metric = (4*math.pi*luas)/(keliling*keliling)
        shape_props=[eccentricity,metric]
        for item in shape_props:
            sheet.write(baris,kolom,item)
            kolom+=1

        # Ekstraksi warna HSV
        hsv = cv2.cvtColor(png, cv2.COLOR_BGR2HSV)
        height=png.shape[0]
        width=png.shape[1]
        H=hsv[:,:,0]
        S=hsv[:,:,1]
        V=hsv[:,:,2]

        hue = np.reshape(H,(1,height*width))
        mode_h = stats.mode(hue[0])
        if int(mode_h[0])==0:
            mode_hue = np.mean(H)
        else:
            mode_hue = int(mode_h[0])

        mean_s = np.mean(S)
        mean_v = np.mean(V)

        color_props=[mode_hue,mean_s,mean_v]
        sheet.write(baris,kolom,mode_hue)
        kolom+=1
        sheet.write(baris,kolom,mean_s)
        kolom+=1
        sheet.write(baris,kolom,mean_v)
        kolom+=1

        sheet.write(baris,kolom,i)
        kolom+=1
        baris+=1
dataset_excel.close()

input("\nLanjut? [enter] ")

import main