# Este es un programa para ejecutar la extensión de red neuronal de la regresión de Cox
#Clase:
#   CoxMlp - clase contenedora para la capa de salida y la capa oculta

#Funciones: 
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

        self.n_input = n_input
        self.n_hidden = n_hidden
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

# Función para dar valores por defecto a los parámetros de la validación cruzada
def defineCVParams(cv_params):

    cv_seed = cv_params.get('cv_seed', 1)
    n_folds = cv_params.get('n_folds', 10) 
    cv_metric = cv_params.get('cv_metric', "loglikelihood") 
    search_iters = cv_params.get('search_iters', 3) 
    L2_range = cv_params.get('L2_range', [-5,-1])

    return(cv_seed, n_folds, cv_metric, search_iters, L2_range)

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

# Función necesaria para evaluar el modelo en conjuntos de prueba
def predictNewData(model, x_test, device = "cpu"):

    device = device if torch.accelerator.is_available() else "cpu"

    x_test_tensor = torch.as_tensor(x_test, dtype=torch.float32).to(device)

    model.eval()

    with torch.no_grad():
        pred = model(x_test_tensor)
    
    np_pred = pred.numpy(force=True)

    return np_pred.flatten()

# Mide la verosimilitud logarítmica, métrica para identificar la calidad predictiva del modelo
def CVLoglikelihood(model, x_full, ytime_full, ystatus_full, x_train, ytime_train, ystatus_train, device="cpu"):

    device = device if torch.accelerator.is_available() else "cpu"

    # Para poder tener los datos de entrenamiento y validación juntos ordenados de mayor a menor supervivencia
    ystatus_full_ordered, ytime_full_ordered, x_full_ordered = data_loader(x_train=x_full,ytime_train= ytime_full,ystatus_train= ystatus_full, device=device)

    pred_full = predictNewData(model=model,x_test=x_full_ordered, device=device)

    pred_full = torch.tensor(pred_full)

    exp_pred_full = torch.exp(pred_full)

    risk_sum = torch.flip(torch.cumsum(torch.flip(exp_pred_full, dims=[0]), dims=[0]), dims=[0])

    log_acc_sum = torch.log(risk_sum)

    subtraction = torch.sum((pred_full - log_acc_sum) * ystatus_full_ordered)

    ystatus_train_ordered, ytime_train_ordered, x_train_ordered = data_loader(x_train=x_train,ytime_train= ytime_train,ystatus_train= ystatus_train, device=device)
    
    pred_train = predictNewData(model=model,x_test=x_train_ordered, device=device)

    pred_train = torch.tensor(pred_train)

    exp_pred_train = torch.exp(pred_train)

    risk_sum = torch.flip(torch.cumsum(torch.flip(exp_pred_train, dims=[0]), dims=[0]), dims=[0])

    log_acc_sum = torch.log(risk_sum)

    subtraction2 = torch.sum((pred_train - log_acc_sum) * ystatus_train_ordered)
    
    return (subtraction-subtraction2)

# Función para calcular el índice C, prácticamente igual que la antigua
def CIndex(model, x_test, ytime_test, ystatus_test, device = "cpu"):

    device = device if torch.accelerator.is_available() else "cpu"

    concord = 0.0
    total = 0.0
    N_test = int(ystatus_test.shape[0])

    ystatus_test = np.asarray(ystatus_test, dtype=bool)
    ytime_test = np.asarray(ytime_test)
    theta = predictNewData(model=model,x_test=x_test,device=device)

    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]: concord = concord + 1
                    elif theta[j] == theta[i]: concord = concord + 0.5

    return(concord/total)

# Función para hacer la validación cruzada que se usará en L2CVProfile y en L2CVSearch
def crossValidate(x_train, ytime_train, ystatus_train, n_hidden,l2=np.exp(-1), search_params = dict(),cv_params = dict(), device="cpu"):

    device = device if torch.accelerator.is_available() else "cpu"

    cv_seed, n_folds, cv_metric, search_iters, L2_range = defineCVParams(cv_params)

    N_train = int(ytime_train.shape[0])
    cv_likelihoods = np.zeros([n_folds], dtype=np.dtype("float64"))
    cv_folds= ms.KFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    k=0

    for traincv, testcv in cv_folds.split(x_train):

        x_train_cv = x_train[traincv]
        ytime_train_cv = ytime_train[traincv]
        ystatus_train_cv = ystatus_train[traincv]

        model, loss_values = trainCoxMLP(x_train = x_train_cv, ytime_train = ytime_train_cv, ystatus_train = ystatus_train_cv, n_hidden=n_hidden,l2=l2, search_params = search_params,device=device,graphic=False)
        
        x_test_cv = x_train[testcv]
        ytime_test_cv = ytime_train[testcv]
        ystatus_test_cv = ystatus_train[testcv]
        
        if cv_metric == "loglikelihood":
            cv_likelihoods[k] = CVLoglikelihood(model=model,x_full=x_train,ytime_full=ytime_train,ystatus_full=ystatus_train,x_train= x_train_cv,ytime_train= ytime_train_cv,ystatus_train= ystatus_train_cv,device=device)
        else:
            cv_likelihoods[k] = CIndex(model=model,x_test=x_test_cv,ytime_test= ytime_test_cv,ystatus_test= ystatus_test_cv, device=device)
        k += 1     
        
    return(cv_likelihoods)

# Función para obtener el mejor L2 de entre el rango que se le debe pasar
def L2CVSearch(x_train, ytime_train, ystatus_train, n_hidden,search_params = dict(),cv_params = dict(),device="cpu"):
    
    device = device if torch.accelerator.is_available() else "cpu"

    cv_seed, n_folds, cv_metric, search_iters, L2_range = defineCVParams(cv_params)
    
    N_train = int(ytime_train.shape[0])
    step_size = float(abs(L2_range[1] - L2_range[0]) / 2)
    L2_reg = float(L2_range[0] + L2_range[1]) / 2
    cv_likelihoods = np.zeros([0, n_folds], dtype=float)
    L2_reg_params = np.zeros([0], dtype="float")
    mean_cvpl = np.zeros([0], dtype="float")
    #best_L2s = np.zeros([0], dtype="float")
    
    l2 = np.exp(L2_reg)
    cvpl = crossValidate(x_train, ytime_train, ystatus_train,n_hidden=n_hidden, l2=l2, search_params=search_params,cv_params= cv_params, device=device)
    cv_likelihoods = np.concatenate((cv_likelihoods, [cvpl]), axis=0)
    L2_reg_params = np.append(L2_reg_params,L2_reg)
    mean_cvpl = np.append(mean_cvpl,np.mean(cvpl))
    best_cvpl = np.mean(cvpl)
    best_L2 = L2_reg

    for i in range(search_iters):
        step_size = step_size/2
        #right
        l2 = np.exp(best_L2 + step_size)
        right_cvpl = crossValidate(x_train, ytime_train, ystatus_train,n_hidden=n_hidden, l2=l2, search_params=search_params,cv_params= cv_params, device=device)
        cv_likelihoods = np.concatenate((cv_likelihoods, [right_cvpl]), axis=0)
        L2_reg_params = np.append(L2_reg_params,best_L2 + step_size)
        mean_cvpl = np.append(mean_cvpl,np.mean(right_cvpl))
        #left
        l2 = np.exp(best_L2 - step_size)
        left_cvpl = crossValidate(x_train, ytime_train, ystatus_train,n_hidden=n_hidden, l2=l2, search_params=search_params,cv_params= cv_params, device=device)
        cv_likelihoods = np.concatenate((cv_likelihoods, [left_cvpl]), axis=0)
        L2_reg_params = np.append(L2_reg_params,best_L2 - step_size)
        mean_cvpl = np.append(mean_cvpl,np.mean(left_cvpl))
        
        if np.mean(right_cvpl) > best_cvpl or np.mean(left_cvpl) > best_cvpl:
            if np.mean(right_cvpl) > np.mean(left_cvpl):
                best_cvpl = np.mean(right_cvpl)
                best_L2 = best_L2 + step_size
            else:
                best_cvpl = np.mean(left_cvpl)
                best_L2 = best_L2 - step_size

    idx = np.argsort(L2_reg_params)
    return(cv_likelihoods[idx], L2_reg_params[idx], mean_cvpl[idx])

# Función para obtener el mejor l2
def L2CVProfile(x_train, ytime_train, ystatus_train, n_hidden,search_params = dict(),cv_params = dict(), device="cpu"):
    
    device = device if torch.accelerator.is_available() else "cpu"

    cv_seed, n_folds, cv_metric, search_iters, L2_range = defineCVParams(cv_params)

    N_train = int(ytime_train.shape[0])
    
    cv_likelihoods = np.zeros([len(L2_range), n_folds], dtype=float)
    mean_cvpl = np.zeros(len(L2_range), dtype="float")
    
    for i in range(len(L2_range)):
        l2 = np.exp(L2_range[i])
        cvpl = crossValidate(x_train, ytime_train, ystatus_train,n_hidden=n_hidden, l2=l2, search_params=search_params,cv_params= cv_params, device=device)
        
        cv_likelihoods[i] = cvpl
        mean_cvpl[i] = np.mean(cvpl)
        
    return(cv_likelihoods, L2_range, mean_cvpl)

# Función para obtener el mejor L2, pero compara más manualmente
def L2Profile(x_train, ytime_train, ystatus_train, x_validation, ytime_validation, ystatus_validation, n_hidden,search_params = dict(),cv_params = dict(), device="cpu"):
    
    device = device if torch.accelerator.is_available() else "cpu"

    cv_seed, n_folds, cv_metric, search_iters, L2_range = defineCVParams(cv_params)
    N_train = int(ytime_train.shape[0])
    
    likelihoods = []

    x_full=np.concatenate([x_train, x_validation], axis=0)
    ytime_full=np.concatenate([ytime_train, ytime_validation])
    ystatus_full=np.concatenate([ystatus_train, ystatus_validation])
    
    for i in range(len(L2_range)):

        l2 = np.exp(L2_range[i])
        model, loss_values = trainCoxMLP(x_train = x_train, ytime_train = ytime_train, ystatus_train = ystatus_train, n_hidden = n_hidden,l2=l2, search_params = search_params, device=device)

        if cv_metric == "loglikelihood":
            likelihoods.append(CVLoglikelihood(model=model, x_full=x_full, ytime_full=ytime_full, ystatus_full=ystatus_full, x_train=x_train, ytime_train=ytime_train, ystatus_train=ystatus_train,device=device))
        else:
            likelihoods.append(CIndex(model= model, x_test=x_validation,ytime_test= ytime_validation,ystatus_test= ystatus_validation,device=device))
        
    return(likelihoods, L2_range)