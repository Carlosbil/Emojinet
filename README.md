# Emojinet
Red neuronal capaz de detectar si la imagen es un emoticono, con interfaz gráfica
# IMPORTANTE
SI SE DESEA USAR CUDA PARA ACELERAR LOS ENTRENAMIENTOS Y TESTS, INTRODUCIR LAS SIGUIENTES LINEAS EN LA 77 Y 107, DENTRO DE LOS BUCLES for batch, (X,Y)...:
- X = X.cuda()
- y = y.cuda()
- miRed.to(device)
        

# Resumen
Este sistema consiste en un Front-end desarrollado en React que contiene una interfaz gráfica simple 
que permite al usuario introducir una imagen en base64. Además contiene un Back-end desarrollado en
python con una red de neuronas que se encarga de procesar la imagen y detectar si hay o no un emoticono 
en ella

# EmojiNet
Es el apodo dado a la red neuronal.
Consta de una red neuronal, sobre la que se ha hecho Transfer Learning de GoogleNet. Se ha escogido esta 
red neuronal para realizar Transfer Learning debido a que esta especializada en el reconocimiento de objetos en 
imágenes. 
Para adaptarlo al proyecto, se ha modificado la capa de salida a solo 2 neuronas, es decir; para que solo de 2
respuestas, si es o no un emoticono

# API-REST
La interfaz se comunica con la red neuronal mediante API-REST, utilizando un método POST en /emojinet.
De esta manera el sistema se puede escalar todo lo necesario


