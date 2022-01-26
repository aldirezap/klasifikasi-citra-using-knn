import pandas as pd
import math as mt
import random

print("\nKNN Euclidean Distance ")
k = int(input('\nMasukkan Jumlah K = '))

print("\nLoading Testing, Please Wait...")

# Memanggil Dataset Train
data_test = pd.read_excel('dataset/dataset_knn.xlsx')

# Memasukkan fitur ke list
i = 0
dataset = []
while i < len(data_test):
    dataset.append([
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

# Shuffle Dataset
random.seed(5)
random.shuffle(dataset)

# Buat Data Train & Data Test
i = 0
data_train, data_test = [],[]
while i < len(dataset):
    if i < 234:              # Jumlah Data Uji
        data_test.append([
            dataset[i][0],  dataset[i][1],  dataset[i][2],  dataset[i][3],  dataset[i][4],
            dataset[i][5],  dataset[i][6],  dataset[i][7],  dataset[i][8],  dataset[i][9],
            dataset[i][10], dataset[i][11], dataset[i][12], dataset[i][13], dataset[i][14],
            dataset[i][15], dataset[i][16], dataset[i][17], dataset[i][18], dataset[i][19],
            dataset[i][20], dataset[i][21], dataset[i][22], dataset[i][23], dataset[i][24],
            dataset[i][25], dataset[i][26], dataset[i][27], dataset[i][28], dataset[i][29],
            dataset[i][30] 
            ])

    else :
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

list_pred = []
loop = 0
while loop < len(data_test):

    print("Data ke -",loop+1," = ",data_test[loop][0])

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
    print("Label    = ",uji_label)
    print("Prediksi = ",predict,"\n")

    # Memasukkan ke list pred untuk di validasi dengan label sebenarnya
    list_pred.append(predict)

    loop += 1  

# Validasi list_pred dengan label
i = 0
tot_benar = 0
print("\n=== Validasi Prediksi ===")
while i < len(data_test):
    if list_pred[i] == data_test[i][30]:
        tot_benar += 1
        print(i+1,") ",list_pred[i]," | ",data_test[i][30]," = Benar")
    else:        
        print(i+1,") ",list_pred[i]," | ",data_test[i][30]," = Salah")
    i+=1

# Menghitung Akurasi
akurasi = (tot_benar/len(data_test))*100

print("\nNilai K        = ",k)
print("Total Data Uji = ",len(data_test))
print("Total Benar    = ",tot_benar)
print("Akurasi        = ",round(akurasi,3),"%")


input("\nLanjut? [enter] ")

import main
