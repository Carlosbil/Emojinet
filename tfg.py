#inicio

#Librerias para el servidor
from flask import Flask, request,jsonify
from flask_cors import CORS

#LIBRERIAS NECESARIAS PARA IMPORTAR EL DATASET
import torch
import os
from torchvision.io import read_image
import torchvision.transforms as trans
from customDataSet import MiDataSet
from torch.utils.data import DataLoader

#LIBRERIAS NECESARIAS PARA el PIPELINE
import torch.nn as nn
import torch.nn.functional as F

#Libreria de googlenet
import torchvision.models.googlenet as GoogleNet

#Libreria para convertir base64 a imagen y mostrarla
import base64
from PIL import Image
import matplotlib.pyplot as plt
import io

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(f"Using {device} device")

#Transformamos las imagenes  a 32 pixeles y a tensores para pasarlas a la red 
imageTransform = trans.Compose([
    trans.Resize(32),
    trans.ToTensor(),
])
#####CREAR MI PROPIO DATASET 
dataset = MiDataSet(csv_file = 'emoji_train.csv', img_dir = 'CSV/grande/archive/image' , transform = imageTransform,  target_transform =trans.ToTensor(),train=True)

#aux = dataset.__getitem__(131)
#aux2 = dataset.__getitem__(131)

test_dataset = MiDataSet(csv_file = 'emoji_test.csv', img_dir = 'CSV/grande/archive/image' , transform = imageTransform,  target_transform =trans.ToTensor(),train=False)
#Creamos el training y test data
#los batch , son subgrupos de imagenes que toma del data_set 
train_dataloader = DataLoader(dataset, batch_size=317, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=91, shuffle=True)

# el tamaño será la cantidad de batchs que ha creado al particionar el data set en los batch_size
dataset_sizes = {
    'train' : len(train_dataloader),
    'test' : len(test_dataloader)
}
print(dataset_sizes['train'])

#ya lo tenemos pre-entrenado
miRed = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
#para los entrenados no se calcula los gradientes de nuevo
for param in miRed.parameters():
    param.requires_grade = False

#usaremos 1024 nodos de entrada 32x32 pixeles , y 2 de salida , Si es o no un emoticono 
miRed.fc = nn.Linear(in_features=1024, out_features=2, bias=True)
if(os.path.exists('EmojiNet_entrenada.pth')):
    miRed.load_state_dict(torch.load('EmojiNet_entrenada.pth'))
    miRed.eval()
print(miRed)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(miRed.parameters(), lr=1e-3)

#Train loop para entrenar a la red que es lo que tiene que buscar, es decir los emoticonos
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    val_acc= 0
    #X imagenes Y labels
    for batch, (X, y) in enumerate(dataloader):
        # Prediccion y ajustes 
        pred = model(X)
        loss = loss_fn(pred, y)

        # Realizamos Backpropagation para mejorar los resultados
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = pred.data.max(dim=1,keepdim=True)[1]
        aux1= torch.argmax(y,dim=1,keepdim=True)
        val_acc += preds.eq(aux1).sum()
    
        ronda=batch+1
        loss, current = loss.item(), batch * len(X) + 317
        print(f"batch:{ronda:>2d} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    val_acc = val_acc/len(dataloader.dataset) * 100
    print(f"\nTrain Final calification\n  loss: {loss:>7f}  Precisión=  [{val_acc:>5f}]")



def test_loop(dataloader, model, loss_fn):
    model.eval()
    val_acc_anterior = 93
    val_acc = 0
    for batch, (X, y) in enumerate(dataloader):
        # Prediccion y ajustes 
        pred = model(X)
        loss = loss_fn(pred, y)

        preds = pred.data.max(dim=1,keepdim=True)[1]
        aux1= torch.argmax(y,dim=1,keepdim=True)
        val_acc += preds.eq(aux1).sum()
    
    val_acc = val_acc/len(dataloader.dataset) * 100
    #De este modo guardamos siempre el mejor modelo
    if(val_acc > val_acc_anterior):
        val_acc_anterior=val_acc
        torch.save(miRed.state_dict(),'EmojiNet_entrenada.pth')

    print(f"Test calification \n  loss: {loss:>7f}  Precisión=  [{val_acc:>5f}]")


for epoch in range(1,5):
    print()
    print(f"Epoch {epoch}\n-------------------------------")
    train_loop(train_dataloader,miRed,loss_fn,optimizer)
    test_loop(test_dataloader,miRed,loss_fn)

#clase para levantar el servidor y crear la api rest
imagen64=""
app = Flask(__name__)
#aceptamos los cors para poder comunicarnos con react /// MOVER a otro archivo .py  todo esto del server?
cors = CORS(app, resources={r"*": {"origins": "*"}})
class servidor:
    def inicio(self):
        print("hola mundo")
    #metodo post que recogera la imagen que queremos saber si tiene o no un emoticono
    @app.route('/',methods=['POST'])
    def ruta():
        imagen= request.get_json()
        imagen64=imagen["imagen"]
        print(imagen64)
        return jsonify(hola = "hola")

    @app.route('/emojinet',methods=['POST'])
    def emojiNet():
        recibido= request.get_json()
        imagen64=recibido["imagen"]
        imagen = base64.b64decode(imagen64)
        resultado=pruebaImagen(miRed,imagen,imageTransform)
        if(resultado==0):
            respuesta= "Es un emoticono"
        else:
            respuesta= "No es un emoticono"
        print(respuesta)
        return jsonify(respuesta = respuesta)

def pruebaImagen (model,imagen,tranform):
    model.eval()
    imagen = Image.open(io.BytesIO(imagen)).convert('RGB')
    imagen_tensor = tranform(imagen)
    imagen_batch = imagen_tensor.unsqueeze(0)
    pred= miRed(imagen_batch)
    resultado = pred.data.cpu().numpy().argmax()
    print(resultado)
    return resultado

imagen64 = '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'
imagen = base64.b64decode(imagen64)
pruebaImagen(miRed,imagen,imageTransform)
#Iniciamos el servidor
servidor = servidor()
#NO BORRAR ESTO 
#con esto se levanta el servidor
if __name__ == '__main__':
    app.run()  
