import os # 處理字符串路徑
import glob # 查找文件
#加載數據
import os
from PIL import Image
import numpy as np
#讀取文件夾PetImages下的2000張圖片，圖片為彩色圖，所以為3通道，
#如果是將彩色圖作為輸入,圖像大小224*224

choose = 1000

def load_data():
    
    data = np.empty((choose*2,224,224,3),dtype="float32")
    label = np.empty((choose*2,))

    # 貓 load 1000張
    
    catimgs = sorted(os.listdir("PetImages/Cat/"))
    catnum = len(catimgs)        
    for i in range(choose):
        cat = Image.open("PetImages/Cat/"+catimgs[i])
        arr = np.asarray(cat, dtype="float32")
        arr.resize((224,224,3))
        data[i, :, :, :] = arr

        label[i] = 0

    # 狗 load 1000張
    dogimgs = sorted(os.listdir("PetImages/Dog/"))
    dognum = len(dogimgs)
    for i in range(choose):
        dog = Image.open("PetImages/Dog/"+dogimgs[i])
        arr = np.asarray(dog, dtype="float32")
        arr.resize((224,224,3))
        data[i+choose, :, :, :] = arr

        label[i+choose] = 1
    
    return data,label

if __name__ == "__main__":
    data,label = load_data()
    np.savez('Cat_Vs_Dogs_1000',data=data,label=label)
    # Cat_Vs_Dogs = np.load('Cat_Vs_Dogs.npz')
    # data = Cat_Vs_Dogs['data']
    # label = Cat_Vs_Dogs['label']
    # print(label.shape)
    print('ok')
