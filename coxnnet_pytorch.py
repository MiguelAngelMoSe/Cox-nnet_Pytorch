# Este es un programa para ejecutar la extensión de red neuronal de la regresión de Cox
#Clase:
#   CoxMlp - clase contenedora para la capa de salida y la capa oculta

#Funciones: 
#   createSharedDataset - función auxiliar para crear un conjunto de datos compartido en Theano (sujeta a cambios de Pytorch)
#   trainCoxMlp - función principal para entrenar el modelo cox-nnet
#   predictNewData - función para predecir nuevos datos
#   L2CVSearch - función auxiliar para realizar validación cruzada en un conjunto de entrenamiento, para seleccionar el parámetro de regularización L2 óptimo.
#   CVLoglikelihood - calcula la probabilidad de validación cruzada (utilizando el método de Houwelingen et al. 2005)
#   varImportance - calcula la importancia de las variables (utilizando el método de Fischer 2015)
#   saveModel - guarda un modelo en un fichero: saveModel(model, file_name)
#   loadModel - carga un modelo desde un fichero: loadModel(fileName)

# Librerías indispensables en la implementación anterior, actualizadas
import time
import numpy as np
import random
import torch
from torch import nn
from sklearn import model_selection as ms

# Constante global para especificar si usamos la gpu como aceleradora o en el caso de que no haya una disponible usar cpu
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# Para que los pesos sean los mismos siempre al empezar, siempre se hace globalmente
torch.manual_seed(123)

class CoxMLP(nn.Module):
    # Aquí se definen las capas de la NN
    # Necesitaremos: tamaño de entrada/número de columnas y (opcional) número neuronas capa oculta
    def __init__(self, n_input, n_hidden):
        super().__init__()

        if n_hidden == None:
            n_hidden = int(np.ceil(n_input ** 0.5))
        
        # Modo manual
        self.input_layer = nn.Linear(n_input,n_hidden)
        self.tanh = nn.Tanh()
        self.output_layer = nn.Linear(n_hidden,1)

    # Y aquí se define como se avanza a través de la NN
    def forward(self, input):

        input_linear = self.input_layer(input)

        linear_tanh = self.tanh(input_linear)

        tanh_output = self.output_layer(linear_tanh)

        return tanh_output
