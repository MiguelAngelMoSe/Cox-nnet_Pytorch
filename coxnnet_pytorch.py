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
import matplotlib.pyplot as plt
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

    cox_pred = cox_pred.reshape(-1)
    ystatus = ystatus.reshape(-1)

    exp_cox_pred = torch.exp(cox_pred)

    risk_sum = torch.flip(torch.cumsum(torch.flip(exp_cox_pred, dims=[0]), dims=[0]), dims=[0])

    log_acc_sum = torch.log(risk_sum)

    subtraction = (cox_pred - log_acc_sum) * ystatus

    return -torch.mean(subtraction)

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
def trainCoxMLP(x_train, ytime_train, ystatus_train, n_hidden, l2=np.exp(-1), search_params = dict(), device="cpu",graphic=False):

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

    # Se configura el optimizador, encargado de actualizar los parámetros basándose en los gradientes
    if method == "momentum":
        optimizer = torch.optim.SGD(params=model.parameters(),lr=l_rate,momentum=momentum,weight_decay=L2_reg)
        print("Using momentum gradient")

    elif method == "nesterov":
        optimizer = torch.optim.SGD(params=model.parameters(),lr=l_rate,momentum=momentum,weight_decay=L2_reg,nesterov=True)
        print("Using nesterov accelerated gradient")

    else:
        optimizer = torch.optim.SGD(params=model.parameters(),lr=l_rate,momentum=0,weight_decay=L2_reg)
        print("Using gradient descent")

    best_loss = float('inf')
    current_lr = l_rate
    loss_values = []
    start = time.time()

    print("Training model")
    for epoch in range(max_iter):

        # Pasos a realizar en el entrenamiento, reiniciamos los gradientes
        optimizer.zero_grad()
    
        # Extraemos predicción y el resultado de la loss function
        pred = model(x_train_ordered)
        loss = negative_log_likelihood(cox_pred=pred, ystatus=ystatus_train_ordered)

        # Se propaga el error hacia atrás y se actualizan los pesos
        loss.backward()
        optimizer.step()

        # Guardamos los valores para mostrarlos si quiere el usuario con graphic=True
        loss_values.append(loss.item())

        if epoch % eval_step == 0:

            current_loss = loss.item()

            if current_loss > best_loss:

                current_lr *= lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:

                current_lr *= lr_growth
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            if current_loss < best_loss * stop_threshold:
                best_loss = current_loss
                patience = max(patience, epoch * patience_incr)
            
            if epoch >= patience:
                print(f"Early stopping in epoch: {epoch}")
                break


    print(('running time: %f seconds') % (time.time() - start))
    print(('total epochs: %f') % (epoch))

    if graphic:
        fig, ax = plt.subplots(figsize=(8,5))
        plt.plot(loss_values)
        plt.title("Step-wise Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
    
    return(model, loss_values)

