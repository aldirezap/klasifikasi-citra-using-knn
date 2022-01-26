import os
def knn_euclidean():
    os.system('knn_euclidean.py')
def test_euclidean():
    os.system('test_euclidean.py')

def knn_manhattan():
    os.system('knn_manhattan.py')
def test_manhattan():
    os.system('test_manhattan.py')

def preprocessing():
    os.system('preprocessing.py')

print("\n-------------- PILIH PROSES --------------")
print("1. Preprocessing")
print("2. Accuracy Euclidean Distance")
print("3. Accuracy Manhattan Distance")
print("4. Test Image Euclidean Distance")
print("5. Test Image Manhattan Distance")
print("6. Exit")
x = int(input("Masukkan Pilihan : "))

if x == 1:
    preprocessing()
elif x == 2:
    knn_euclidean()
elif x == 3:
    knn_manhattan()
elif x == 4:
    test_euclidean()
elif x == 5:
    test_manhattan()
else:
    print("\nSelesai")
