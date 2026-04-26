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
    
# Loss function en proceso, en este caso la función de negative_log_likelihood
# cox_pred es el riesgo que nos devuelve el forward y ystatus es la variable estado de los datos, referida ha si han sufrido evento o no
def negative_log_likelihood(cox_pred, ystatus):
        
    exp_cox_pred = torch.exp(cox_pred)

    accumulated_sum = torch.cumsum(torch.inverse(exp_cox_pred))

    log_acc_sum = torch.log(accumulated_sum)

    subtraction = cox_pred - log_acc_sum

    product_status = subtraction * ystatus

    return -torch.mean(product_status)

# Función para ordenar los datos de supervivencia de mayor supervivencia a menor para la loss function
# Necesita los datos y un parámetro opcional device que usará cpu por defecto o el acelerador que se le pase
def data_loader(x_train,ytime_train,ystatus_train, device="cpu"):

    ytime_tensor = torch.from_numpy(ytime_train).float()
    ystatus_tensor = torch.from_numpy(ystatus_train).float()
    x_tensor = torch.tensor(x_train).float()

    time_index = torch.argsort(ytime_tensor,descending=True)

    ystatus_train_ordered = ystatus_tensor[time_index]
    ytime_train_ordered = ytime_tensor[time_index]
    x_train_ordered = x_tensor[time_index]

    return ystatus_train_ordered.to(device), ytime_train_ordered.to(device), x_train_ordered.to(device)

# Función para darle valores por defecto a la configuración del entrenamiento, igual que la antigua versión
def defineSearchParams(search_params):

    method = search_params.get('method', "nesterov")
    learning_rate = search_params.get('learning_rate', 0.01)
    momentum = search_params.get('momentum', 0.9) 
    lr_decay = search_params.get('lr_decay', 0.9)
    lr_growth = search_params.get('lr_growth', 1.0)
    eval_step = search_params.get('eval_step', 23)
    max_iter = search_params.get('max_iter', 10000)
    stop_threshold = search_params.get('stop_threshold', 0.995)
    patience = search_params.get('patience', 2000)
    patience_incr = search_params.get('patience_incr', 2)
    rand_seed = search_params.get('rand_seed', 123) 

    return(method, learning_rate, momentum, lr_decay, lr_growth, eval_step, max_iter, stop_threshold, patience, patience_incr, rand_seed)

# Función de entrenamiento
def trainCoxMLP(x_train, ytime_train, ystatus_train, n_hidden, l2=np.exp(-1), search_params = dict(), device="cpu", verbose=False):

    device = device if torch.accelerator.is_available() else "cpu"
    L2_reg = l2
    method, l_rate, momentum, lr_decay, lr_growth, eval_step, max_iter, stop_threshold, patience, patience_incr, rand_seed = defineSearchParams(search_params)
    # Para que los pesos sean los mismos siempre al empezar
    torch.manual_seed(rand_seed)

    # Para darle al modelo un número de neuronas de entrada igual al número de columnas del dataset
    n_train = x_train.shape[1]


    ystatus_train_ordered, ytime_train_ordered, x_train_ordered = data_loader(x_train, ytime_train, ystatus_train, device)

    # Creamos el modelo y lo pasamos a device
    model = CoxMLP(n_train, n_hidden).to(device)

    #cost = (negative_log_likelihood(model.forward(x_train_ordered), ystatus_train_ordered) + L2_reg * model.L2_sqr)

    # Arreglar esta zona
    if method == "momentum":
        updates = torch.optim.SGD(model.params, l_rate, momentum)
        print("Using momentum gradient")
    elif method == "nesterov":
        updates = torch.optim.SGD(model.params,nesterov=True, l_rate, momentum)
        print("Using nesterov accelerated gradient")
    else:
        updates = torch.optim.SGD(model.params, l_rate, 0)
        print("Using gradient descent")
    return