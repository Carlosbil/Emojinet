import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
import base64
from PIL import Image
import io
import torchvision.transforms as trans
class MiDataSet(Dataset):

#csv_file  ( mi archivo csv)
#img_dir (ruta de las imagenes)
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None, train=True):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.datasets = 0
        self.train=train
        self.cifar= torchvision.datasets.CIFAR10(root='./data', train=self.train,
                                        download=True)

    def __len__(self):
        return len(self.img_labels)

    #Combinamos en el dataset imagenes de Emoticonos con los de Cifar10
    def __getitem__(self, idx):
        #EMOJI
        if self.datasets == 0 :
        #así la siguiente vez sera una imagen del CIFAR10
         self.datasets = 1 
         img_path = self.img_labels.iloc[idx, 1] #carga la imagen de apple

        #sacamos el base64 del csv y lo pasamos a imagen
         image = base64.b64decode(img_path)
         image = Image.open(io.BytesIO(image)).convert('RGB')

         plt.imshow(image)
        #aplicamos la transformaciones 
         if self.transform:
            image = self.transform(image)
         if self.target_transform:
            #label = self.target_transform(label)
            label=torch.tensor([1.0,0.0])
         return image, label

        #CIFAR10
        else :
            #así la siguiente vez sera una imagen de los EMOJIS
            self.datasets = 0
            data = self.cifar[idx]
            plt.imshow(data[0])
            if self.transform:
             image = self.transform(data[0])
            if self.target_transform:
             label=torch.tensor([0.0,1.0])
            return image, label
    def transformImage(self,imagen):
        if self.transform:
            image = self.transform(image)
        return imagen
#Función para mostrar la imagen del cifar10
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    print('A')
