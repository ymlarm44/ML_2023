# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:24:25 2022

@author: je_su
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys


def plot_decision_regions(X, y, clasificador, test_idx=None, resolution=0.02):
    
    # marcadores y mapas de colores
    markers = ('s', '^', 'o', 'x', 'v')
    colors = ('red', 'lightgreen', 'blue','gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # superficie de decisión
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z = clasificador.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #Grafica todas las muestras según su clase con distintos
    #marcadores y colores
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black'
                   )

    # resalta las muestras de prueba
    if np.any(test_idx):        
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='None',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')
        

       
        
from numpy.random import RandomState   

class Perceptron(object):
    """Perceptrón simple.
    
    Parámetros
    ------------
    eta : float
        factor de entrenamiento (entre 0.0 y 1.0)
    n_iter : int
        iteraciones para el entrenamiento.
    random_state : int
        Semilla generadora de números aleatorios para la inicialización de los pesos.
    
    Atributos
    -----------
    w_ : 1d-array
        Pesos despues del entrenamiento.
    mal_clasificados_ : list
        Número de desaciertos en cada época
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Función de entrenamiento.
        Parameters
        ----------
        X : array, shape = [n_muestras, n_caracteristicas]
          vector de entrenamiento
        y : array, shape = [n_muestras]
          vector target.
          
        Returns
        -------
        self : objeto

        """        
        rgen = RandomState(self.random_state)
        
        #inicializo los pesos con valores aleatorios entre 0 y 1
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size= 1 + X.shape[1])
        #self.w_ = np.ones((1 + X.shape[1],1)).flatten()
        
        self.mal_clasificados_ = []
        self.errores_ = []
        
        
        for _ in range(self.n_iter):
            
            mal_clasificados = 0
            
            for xi, target in zip(X, y):
                
                # cálculo de la salida 
                y_salida = self.predict(xi)
                # --------cálculo del error y actualizo del vector de pesos-------
                error = target - y_salida                
                update = self.eta * (error)
                self.w_[1:] += update * xi
                self.w_[0] += update
                self.errores_.append(error)
                #------------------------------------------------------------------------
                #si error es distinto de 0 la salida predicha difiere de la salida esperada
                mal_clasificados += int(error != 0.0)
                
            self.mal_clasificados_.append(mal_clasificados)
            
        return self

    def calcular_entrada(self, X):
        """cálculo de la entrada al perceptrón"""
        # -------suma de los productos de los valores de entrada y los pesos -----------
        
        return np.dot(X, self.w_[1:]) + self.w_[0]
        #-------------------------------------------------------------------------------

    def predict(self, X):
        """devuelve la etiqueta de la clase pertenciente después de aplicar la fn. de activación"""
        # la función de activación es la función signo: 
        # 0 si el resultado de calcular_entrada < 0
        # 1 si el resultado de calcular_entrada >= 0
        return np.where( self.calcular_entrada(X) >= 0.0, 1, 0 )
        
        
        
class Adaline(object):
    """ADAptive LInear NEuron classifier.
    Parametros
    ------------
    lr : float
        factor de entrenamiento (entre 0.0 y 1.0)
    n_iter : int
        iteraciones para el entrenamiento.
    random_state : int
        Semilla generadora de números aleatorios para
        la inicialización de los pesos.
    
    Atributos
    -----------
    w_ : 1d-array
        Pesos despues de su ajuste.
    costo_ : list
        Función de costo : suma de los errores cuadráticos en cada época
    """
    def __init__(self, lr=0.01, n_iter=50, random_state=1):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Ajuste de los datos de entrenamiento.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_muestras, n_caracteristicas]
          vector de entrenamiento
        y : array-like, shape = [n_muestras]
          vector target.
          
        Returns
        -------
        self : objeto

        """
        rgen = RandomState(self.random_state)
        #inicializo los pesos con valores aleatorios entre 0 y 1
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.costo_ = []

        for _ in range(self.n_iter):
            entrada = self.calcular_entrada(X)
            salida = self.fn_activacion(entrada)
            # El error se calcula como la diferencia entre las salidas esperadas y
            # la salida de la fn. de activación
            errores = (y - salida)              
            # --------actualización del vector de pesos--------------------------------
            self.w_[1:] += self.lr * X.T.dot(errores)
            self.w_[0] += self.lr * errores.sum()
            # función de costo
            costo = (errores**2).sum() / 2.0
            #---------------------------------------------------------------------------
            self.costo_.append(costo)
        return self
    
    def calcular_entrada(self, X):
        """Cálculo de la entrada al Adaline"""
        # -------suma de los productos de los valores de entrada y los pesos -----------        
        return np.dot(X, self.w_[1:]) + self.w_[0]
        #-------------------------------------------------------------------------------
    
    def fn_activacion(self, X): 
        """Aplica la fn. de activación lineal"""
        #Se puede ver que la fn.  de activación es la fn. identidad, vamos a ver que en el
        #caso de la regresión logística (más adelante), usamos una fn. sigmoidea
        return X

    def predict(self, X):
        """devuelve la etiqueta de la clase pertenciente después de aplicar la fn. de activación"""
        return np.where(self.calcular_entrada(X) >= 0.0, 1, -1)

import numpy as np
from numpy.random import RandomState

class PMC():
    """ Clasificador perceptrón multicapas

    Parametros
    ------------
    nro_ocultas : int (por defecto: 5)
        Número de nodos ocultos.
    alpha : float (por defecto: 0.1)
        Coeficiente de momento
    nro_epocas : int (por defecto: 100)
        Número de épocas.
    eta : float (por defecto: 0.01)
        Coeficiente de aprendizaje.
    seed : int (por defecto: 0)
        semilla random para inicializar los pesos y 
        generar un índice aleatorio
    escalado: bool
        para indicar si se escalan los datos previo
        al entrenamiento

    Atributos
    -----------

    """
    def __init__(self, nro_ocultas=5, eta=0.1, alpha=0.1, nro_epocas=100, seed=0, escalado=False):
        
        self.nro_ocultas = nro_ocultas
        self.alpha = alpha
        self.nro_epocas = nro_epocas
        self.random = RandomState(seed)
        self.eta = eta
        self.escalado = escalado
    
    def fn_activacion(self, z):
        """Aplica la fn. de activación logística (sigmoidea)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def propagacion_hacia_adelante(self, X):
        """Función para realizar la propagación hacia adelante"""
        
        # paso 1: obtenemos la entrada a la capa oculta: z_h
        #----------------------------------------------           
        z_h = np.dot(self.w_h, X.T) 
        z_h = z_h[:,np.newaxis]
        #print('z_h: ',z_h.shape)
        #----------------------------------------------
        
        # paso 2: salida de la capa oculta a_h, después de aplicar la función
        # de activación sigmoidea a z_h 
        # agregamos una fila a a_h correspondiente al bias
        #----------------------------------------------
        a_h = self.fn_activacion(z_h)            
        a_h = np.vstack((a_h, np.ones([a_h.shape[1],1], a_h.dtype)))
        #print('a_h: ',a_h.shape)
        #----------------------------------------------
        
        # paso 3: obtenemos la entrada a la capa de salida: z_out 
        #----------------------------------------------
        #print(w_out.shape)
        z_out = np.dot(self.w_out, a_h) 
        #print('z_out: ',z_out.shape)
        #----------------------------------------------
        
        # paso 4: salida a_out después de aplicar la función
        # de activación sigmoidea a z_out
        #----------------------------------------------
        a_out = self.fn_activacion(z_out)
        #print('a_out: ',a_out.shape)
        #----------------------------------------------
        
        return z_h, a_h, z_out, a_out
    
    def fit(self, X_train, y_train):    
        
        X = np.copy(X_train)
        y = np.copy(y_train)
    
        n_clases_target = np.unique(y).shape[0]  # número de clases target
        n_características = X.shape[1]
        n_patrones = X.shape[0]
        
        ###########################
        # inicialización de pesos #
        ###########################    
        # pesos capa de entrada -> capa oculta
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(self.nro_ocultas, n_características + 1))    
        #print('w_h: ', self.w_h.shape)    
        # pesos capa oculta -> capa de salida
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(y.shape[1], self.nro_ocultas + 1))    
        #print('w_out: ', self.w_out.shape)   
    
        # Escalador Min-Max -> escalado en el rango [0,1]
        if self.escalado:
            from sklearn.preprocessing import MinMaxScaler
            sc = MinMaxScaler()
            sc.fit(X)                     # Estima los parámetros de máximo y mínimo
            X_scaled = sc.transform(X)           # Normaliza los datos usando las estimaciones
        else:
            X_scaled = X
        
        ###################
        # agrego bias a X #
        ###################
        X_scaled = np.hstack( (X_scaled,  np.ones( (X_scaled.shape[0],1) , X_scaled.dtype) ) )
        #print("X_scaled después del bias: ", X_scaled.shape)
        
        #lista para acumular los errores
        self.ecm = []
    
        delta_w_h_anterior = np.zeros(self.w_h.shape)
        #print("delta_w_h_anterior: ", delta_w_h_anterior.shape)
        
        delta_w_out_anterior = np.zeros(self.w_out.shape)
        #print("delta_w_out_anterior: ", delta_w_out_anterior.shape)
        #sum_costo=0
        for i in range(self.nro_epocas):
            for j in range(n_patrones):                
        
                # primero obtenemos un índice aleatorio
                ind = int(np.floor(X_scaled.shape[0]*self.random.rand())) #try random.RandomState.randint
                
                z_h, a_h, z_out, a_out = self.propagacion_hacia_adelante(X_scaled[ind, :])
                
                #----------------------------------------------
                # cálculo de error
                error = y_train[ind,:].T - a_out
                #print('error: ', error.shape) 
                costo = (error**2).sum()/(2.0)
                #sum_costo += costo
                self.ecm.append(costo)
                
                ####################
                # Retropropagación #
                ####################            
                grad_out = np.dot( a_out * (1. - a_out) , error)
                grad_h = (a_h * (1. - a_h) ) * np.dot(self.w_out.T, grad_out)
                #print('grad_h: ', grad_h.shape)
                
                delta_w_out = self.eta * grad_out * a_h.T
                #print('delta_Wout: ', delta_w_out.shape)
                delta_w_out = delta_w_out + self.alpha * delta_w_out_anterior;
                
                delta_w_h = self.eta * grad_h * X_scaled[ind, :]
                #print('delta_Wh: ', delta_Wh)
                
                delta_w_h = delta_w_h[0:delta_w_h.shape[0]-1, :]
                #print('delta_Wh: ', delta_Wh.shape)
                delta_w_h = delta_w_h + self.alpha * delta_w_h_anterior;
                
                self.w_out = self.w_out + delta_w_out
                delta_w_out_anterior = delta_w_out
                
                self.w_h = self.w_h + delta_w_h
                delta_w_h_anterior = delta_w_h
             
    
    def predict(self, X_test):
        """
        Predice la etiqueta de clase
        Parametros
        ----------
        X : array, shape = [n_muestras, n_caracteristicas]
            matriz datos de entrada.
        Returns
        -------
        y_pred: array, shape = [n_muestras].
            etiquetas de clases predichas
        """   
        X = np.copy(X_test)
        # Escalador Min-Max -> escalado en el rango [0,1]
        if self.escalado:
            from sklearn.preprocessing import MinMaxScaler
            sc = MinMaxScaler()
            sc.fit(X)                     # Estima los parámetros de máximo y mínimo
            X_scaled = sc.transform(X)           # Normaliza los datos usando las estimaciones
        else:
            X_scaled = X
        
        X_scaled = np.hstack( (X_scaled,  np.ones( (X_scaled.shape[0],1) , X_scaled.dtype) ) )

        y_pred = []
        for ind in range(X_scaled.shape[0]):        
            z_h, a_h, z_out, a_out = self.propagacion_hacia_adelante(X_scaled[ind,:])
            y_pred.append(a_out)
        y_pred = np.vstack(y_pred)
        y_pred = np.where(y_pred<=0.5, 0, 1)
        return y_pred