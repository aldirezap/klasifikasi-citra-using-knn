import cv2
import numpy as np
import xlsxwriter as xls
from skimage.feature import greycomatrix, greycoprops
import math
from scipy import stats
from tkinter import *
from tkinter import filedialog

def browseFiles():
	filename = filedialog.askopenfilename(initialdir = "/",
										title = "Select a File",
										filetypes = (("Text files",
														"*.txt*"),
													("all files",
														"*.*")))
	
	return filename

# Baris citra jenis mangga
dataset_excel = xls.Workbook('hasil/testt.xlsx')
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

kolom=0
file_name = browseFiles()
print("\n== Testing 1 Image Euclidean Distance == ")
print("File = ",file_name)
print("Loading, Please Wait...")
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

print("Preprocessing Selesai!")
print("==================================")

# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================

import pandas as pd
import math as mt
import random

print("\nKNN Euclidean Distance ")
k = int(input('Masukkan Jumlah K = '))

print("\nLoading Testing, Please Wait...")

# Memanggil Dataset Train
data_train = pd.read_excel('dataset/dataset_knn.xlsx')
data_test  = pd.read_excel('hasil/testt.xlsx')

# Memasukkan fitur ke list
i = 0
dataset = []
while i < len(data_train):
    dataset.append([
        data_train["Data Gambar"][i],
        data_train["correlation 0"][i],   data_train["correlation 45"][i],    data_train["correlation 90"][i],     data_train["correlation 135"][i],
        data_train["homogeneity 0"][i],   data_train["homogeneity 45"][i],    data_train["homogeneity 90"][i],     data_train["homogeneity 135"][i],
        data_train["dissimilarity 0"][i], data_train["dissimilarity 45"][i],  data_train["dissimilarity 90"][i],   data_train["dissimilarity 135"][i],
        data_train["contrast 0"][i],      data_train["contrast 45"][i],       data_train["contrast 90"][i],        data_train["contrast 135"][i],
        data_train["energy 0"][i],        data_train["energy 45"][i],         data_train["energy 90"][i],          data_train["energy 135"][i],
        data_train["ASM 0"][i],           data_train["ASM 45"][i],            data_train["ASM 90"][i],             data_train["ASM 135"][i],
        data_train["metric"][i],          data_train["eccentricity"][i],      data_train["hue"][i],                data_train["saturation"][i], 
        data_train["values"][i],          data_train["label"][i]
        ])
    i += 1

i = 0
data_train = []
while i < len(dataset):    
    data_train.append([
        dataset[i][0],  dataset[i][1],  dataset[i][2],  dataset[i][3],  dataset[i][4],
        dataset[i][5],  dataset[i][6],  dataset[i][7],  dataset[i][8],  dataset[i][9],
        dataset[i][10], dataset[i][11], dataset[i][12], dataset[i][13], dataset[i][14],
        dataset[i][15], dataset[i][16], dataset[i][17], dataset[i][18], dataset[i][19],
        dataset[i][20], dataset[i][21], dataset[i][22], dataset[i][23], dataset[i][24],
        dataset[i][25], dataset[i][26], dataset[i][27], dataset[i][28], dataset[i][29],
        dataset[i][30] 
        ])
    i += 1

# ===========================================================================

i = 0
dataset_test = []
while i < len(data_test):
    dataset_test.append([
        data_test["Data Gambar"][i],
        data_test["correlation 0"][i],   data_test["correlation 45"][i],    data_test["correlation 90"][i],     data_test["correlation 135"][i],
        data_test["homogeneity 0"][i],   data_test["homogeneity 45"][i],    data_test["homogeneity 90"][i],     data_test["homogeneity 135"][i],
        data_test["dissimilarity 0"][i], data_test["dissimilarity 45"][i],  data_test["dissimilarity 90"][i],   data_test["dissimilarity 135"][i],
        data_test["contrast 0"][i],      data_test["contrast 45"][i],       data_test["contrast 90"][i],        data_test["contrast 135"][i],
        data_test["energy 0"][i],        data_test["energy 45"][i],         data_test["energy 90"][i],          data_test["energy 135"][i],
        data_test["ASM 0"][i],           data_test["ASM 45"][i],            data_test["ASM 90"][i],             data_test["ASM 135"][i],
        data_test["metric"][i],          data_test["eccentricity"][i],      data_test["hue"][i],                data_test["saturation"][i], 
        data_test["values"][i],          data_test["label"][i]
        ])
    i += 1

i = 0
data_test = []
while i < len(dataset_test):    
    data_test.append([
        dataset_test[i][0],  dataset_test[i][1],  dataset_test[i][2],  dataset_test[i][3],  dataset_test[i][4],
        dataset_test[i][5],  dataset_test[i][6],  dataset_test[i][7],  dataset_test[i][8],  dataset_test[i][9],
        dataset_test[i][10], dataset_test[i][11], dataset_test[i][12], dataset_test[i][13], dataset_test[i][14],
        dataset_test[i][15], dataset_test[i][16], dataset_test[i][17], dataset_test[i][18], dataset_test[i][19],
        dataset_test[i][20], dataset_test[i][21], dataset_test[i][22], dataset_test[i][23], dataset_test[i][24],
        dataset_test[i][25], dataset_test[i][26], dataset_test[i][27], dataset_test[i][28], dataset_test[i][29],
        dataset_test[i][30] 
        ])
    i += 1

list_pred = []
loop = 0
while loop < len(data_test):    

    # Fitur / Class
    uji_name         = data_test[loop][0]
    uji_corel_0      = data_test[loop][1]
    uji_corel_45     = data_test[loop][2]
    uji_corel_90     = data_test[loop][3]
    uji_corel_135    = data_test[loop][4]
    uji_homo_0       = data_test[loop][5]
    uji_homo_45      = data_test[loop][6]
    uji_homo_90      = data_test[loop][7]
    uji_homo_135     = data_test[loop][8]
    uji_dissim_0     = data_test[loop][9]
    uji_dissim_45    = data_test[loop][10]
    uji_dissim_90    = data_test[loop][11]
    uji_dissim_135   = data_test[loop][12]
    uji_contrast_0   = data_test[loop][13]
    uji_contrast_45  = data_test[loop][14]
    uji_contrast_90  = data_test[loop][15]
    uji_contrast_135 = data_test[loop][16]
    uji_energy_0     = data_test[loop][17]
    uji_energy_45    = data_test[loop][18]
    uji_energy_90    = data_test[loop][19]
    uji_energy_135   = data_test[loop][20]
    uji_asm_0        = data_test[loop][21]
    uji_asm_45       = data_test[loop][22]
    uji_asm_90       = data_test[loop][23]
    uji_asm_135      = data_test[loop][24]
    uji_metric       = data_test[loop][25]
    uji_eccentricity = data_test[loop][26]
    uji_hue          = data_test[loop][27]
    uji_saturation   = data_test[loop][28]
    uji_values       = data_test[loop][29]
    uji_label        = data_test[loop][30]    

    # 1. Menghitung Eucledian distance
    knn = []
    i = 0           # inisialisasi iterasi
    while i < len(data_train):
        knn.append([
        data_train[i][0], data_train[i][30], round(mt.sqrt(
                (data_train[i][1]      - uji_corel_0)**2 +
                (data_train[i][2]     - uji_corel_45)**2 +
                (data_train[i][3]     - uji_corel_90)**2 +
                (data_train[i][4]    - uji_corel_135)**2 +
                (data_train[i][5]       - uji_homo_0)**2 +
                (data_train[i][6]      - uji_homo_45)**2 +
                (data_train[i][7]      - uji_homo_90)**2 +
                (data_train[i][8]     - uji_homo_135)**2 +
                (data_train[i][9]     - uji_dissim_0)**2 +
                (data_train[i][10]    - uji_dissim_45)**2 +
                (data_train[i][11]    - uji_dissim_90)**2 +
                (data_train[i][12]   - uji_dissim_135)**2 +
                (data_train[i][13]   - uji_contrast_0)**2 +
                (data_train[i][14]  - uji_contrast_45)**2 +
                (data_train[i][15]  - uji_contrast_90)**2 +
                (data_train[i][16] - uji_contrast_135)**2 +
                (data_train[i][17]     - uji_energy_0)**2 +
                (data_train[i][18]    - uji_energy_45)**2 +
                (data_train[i][19]    - uji_energy_90)**2 +
                (data_train[i][20]   - uji_energy_135)**2 +
                (data_train[i][21]        - uji_asm_0)**2 +
                (data_train[i][22]       - uji_asm_45)**2 +
                (data_train[i][23]       - uji_asm_90)**2 +
                (data_train[i][24]      - uji_asm_135)**2 +
                (data_train[i][25]       - uji_metric)**2 +
                (data_train[i][26] - uji_eccentricity)**2 +
                (data_train[i][27]          - uji_hue)**2 +
                (data_train[i][28]   - uji_saturation)**2 +
                (data_train[i][29]       - uji_values)**2),4)])
        i += 1

    # 2. Mengurutkan hasil jarak terdekat
    knn = sorted(knn,key=lambda l:l[2])

    # Menampilkan n terdekat
    i = 0
    while i < len(knn):
        if i < 15:
            print(knn[i])
        i+=1

    # 3. Menghitung jumlah dari fitur prediksi
    i, aufush, dasheri, jamadar, kesar, rajapuri, totapuri = 0,0,0,0,0,0,0
    while i < k:
        if knn[i][1] == "Aafush":
            aufush += 1
        elif knn[i][1] == "Dasheri":
            dasheri += 1
        elif knn[i][1] == "Jamadar":
            jamadar += 1
        elif knn[i][1] == "Kesar":
            kesar += 1
        elif knn[i][1] == "Rajapuri":
            rajapuri += 1
        elif knn[i][1] == "Totapuri":
            totapuri += 1
        i += 1

    rank = []
    rank.append([aufush,"Aafush"])
    rank.append([dasheri,"Dasheri"])
    rank.append([jamadar,"Jamadar"])
    rank.append([kesar,"Kesar"])
    rank.append([rajapuri,"Rajapuri"])
    rank.append([totapuri,"Totapuri"])
    rank = sorted(rank,key=lambda l:l[0], reverse=True)    

    # 4. Prediksi
    predict = rank[0][1]    
    print("Prediksi = ",predict)

    loop += 1  

input("\nLanjut? [enter] ")

import main
