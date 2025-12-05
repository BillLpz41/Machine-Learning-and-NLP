import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

y_pred=[]

#Clase para el conjunto de entrenamiento
class train_set:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

#Clase para el conjunto de prueba
class test_set:
	def __init__(self, x_test, y_test):
		self.x_test = x_test
		self.y_test = y_test

#Clase para el conjunto de datos final
class final_dset:
    def __init__(self, test_set, train_set):
         self.test_set = test_set
         self.train_set = train_set

#Dividir el conjunto de datos de los emails
def generate_dset(file_name):
	pd.options.display.max_colwidth = 200				

	#Leactura del archivo csv original
	df = pd.read_csv(file_name, sep=',', engine='python')
    #Columnas con los datos a utilizar para la predicion
	x = df.drop(['target'],axis=1).values   
    #Columna a predecir
	y = df['target'].values
	
	#Dividr 70% de los datos para entrenamiento y 30% para pruebas
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle = True, random_state=0)
	
	
	#Crea un test set
	my_test_set = test_set(x_test, y_test)
	#Crea un train set
	my_train_set = train_set(x_train, y_train)
	#Crea un data set con el validation set, test set y train set
	my_data_set = final_dset(my_test_set, my_train_set) 
    
	return (my_data_set)


def hiperplano_heart(data_set):
   
    x_train= data_set.train_set.x_train
    y_train= data_set.train_set.y_train

    x_test= data_set.test_set.x_test
    y_test= data_set.test_set.y_test

    #Cálculo de C+ y C-
    c_pos=0
    c_neg=0

    #Suma de los puntos positivos y negativos
    sum_points_pos=0
    sum_points_neg=0

    #Cantidad de positivos y negativos
    M_pos=0
    M_neg=0

    #Sumatoria para C+ y C-
    for i in range(len(x_train)):
        if y_train[i] == 1:
            #print("positivo")
            sum_points_pos= sum_points_pos + x_train[i]
            M_pos= M_pos + 1
        #if y_train[i] == 0:
        else:
            #print("negativo")
            sum_points_neg= sum_points_neg + x_train[i]
            M_neg= M_neg + 1

    #print(M_pos)
    c_pos= sum_points_pos/M_pos
    print('\nPromedio de positivos (C+)',c_pos)

    #print(M_neg)
    c_neg= sum_points_neg/M_neg
    print('\nPromedio de negativos (C-)',c_neg)
	
    #Cálculo del punto intermedio C
    c= (c_pos+c_neg)/2
    print('\nPunto intermedio C: ', c)
    
    #Calculando la magnitud
    suma_cuadrada=0
    for eje in c:
        suma_cuadrada= suma_cuadrada + (eje**2)
        #print(suma_cuadrada)
	
    cmag= math.sqrt(suma_cuadrada)
    print('\nMagnitud de C: ', cmag)

    #Cálculo de las proyecciones de los datos de prueba
    for point in x_test:
        #print(point)
        proyeccion(point, c, cmag)

    #Cálculo de la exactitud
    accuracyM = accuracy_score(y_test, y_pred)
    print("\nAccuracy medido para el 30% del data set original: {}\n".format(accuracyM))
    
    #Reporta los resultados de la clasificación de los datos de prueba
    print(classification_report(y_test, y_pred, target_names = ['0','1']))
	
	#Crea la matriz de confusión de los datos de prueba
    cm = confusion_matrix(y_test, y_pred)
	
	#Muestra la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
    #print(y_pred)
    #Crea un archivo csv para facilitar la comparación de resultados
    datos=[[],[]]
        
    datos[0]= y_test
    datos[1]= y_pred
    datos=np.column_stack([datos[0], datos[1]])    

    np.savetxt("prediccion_heart.csv", datos, delimiter=",", fmt="%s", header="Prediction, Resultado", comments="")
        
def proyeccion(p_test, p_intermedio, magnitud):
  
    proy= np.dot(p_test, p_intermedio)/magnitud
    #print("La proyección es de: ", proy/magnitud)
    if proy <= magnitud:
        #Pertenece a la clase negativa
        #print("El dato es 0, con proyeccion de {}".format(proy))
        y_pred.append(0)
    else:
        #Pertenece a la clase positiva
        #print("El dato es 1, con proyeccion de {}".format(proy))
        y_pred.append(1)


if __name__=='__main__':
	print("\n---------------Hiperplano con heart---------------\n")
	
	my_dset= generate_dset('heart.csv')
	hiperplano_heart(my_dset)
	
	print("\n---------------Fin del programa---------------\n")
	