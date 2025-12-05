import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import sys

#Funcion de pérdida
def F(w, x, y):
    #a=sum((w * x - y)**2 for x, y in zip(x, y))
    #print('a {}'.format(a))
    #Sumatoria de la función de pérdida, la cual usa a x & y como una tupla del zip(x,y)
    return sum((w * x - y)**2 for x, y in zip(x, y))/len(y)

#Gradiente
def dF(w, x, y):
	return sum(2*(w * x - y) * x for x, y in zip(x, y))/len(y)

def print_line(points, w, iteration, line_color = None, line_style = 'dotted'):  
    list_x=[]
    list_y=[]

    #tuple obtiene una tupla de los puntos (x,y)
    for index, tuple in enumerate(points):
        #x recibe la primera parte de la tupla, la cual es el eje x
        x = tuple[0]
        y = x * w
        #Se añaden los puntos a las listas
        list_x.append(x)
        list_y.append(y)
    plt.text(x,y, iteration, horizontalalignment='right')
    plt.plot(list_x, list_y, color = line_color, linestyle= line_style)

if __name__=='__main__':
    #Lee el corpus original del archivo de entrada y lo pasa a un DataFrame
    df = pd.read_csv('dataset_ejercicio_I_regresion_lineal.csv', sep=',', engine='python')
    #No incluye la columna price
    x = df.drop(['price'],axis=1).values
    #Usa la columna price   
    y = df['price'].values
    #El arreglo x está en 2D (debido a que pudo haber añadido más de una columna)
    #Así que se converte en 1D con la misma longitud de x
    #x= np.reshape(x, len(x))

    #Separa el corpus cargado en el DataFrame con 90% para entrenamiento y 10% para pruebas
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle = True, random_state=0)
    #Recibe la cantidad de iteraciones deseadas (ingresado después del nombre del archivo en la terminal)
    iterations = int(sys.argv[1])
    #Coloca los puntos en la gráfica (gráfico de dispersión)
    plt.scatter(X_train, y_train)

    #print('\n')
    #print ('X = {}\n y = {}'.format(x[2], y[2]))
    #print ('X_train = {}\n y_train = {}'.format(X_train, y_train))
    #print ('X_test = {}\n y_test = {}'.format(X_test, y_test))

    #Train
    print('\nTrain')
    w= 0
    alpha = 0.000008
	# ~ alpha = 0.05 #Efecto similar al de no sacar el promedio
    #Ciclo de contador t con el rango de iteraciones de 5 (de 0 a 4)
    for t in range(iterations):
        loss_function = F(w, X_train, y_train)
        #print('loss_function= {}'.format(loss_function))
        
        #El gradiente será negativo
        gradient = dF(w, X_train, y_train)
        #print('gradient= {}'.format(gradient))
        
        w = w - alpha * gradient
        #print('w= {}'.format(w))
        
        print ('iteration {}: w = {}, F(w) = {}'.format(t, w, loss_function))
        print_line(zip(X_train, y_train), w, t)

    print ('mse: {}'.format(loss_function))
    print_line(zip(X_train, y_train), w, t, 'red', 'solid')
    plt.show()


	#Test
    print('\nTest')
    print ('Calculated weight: {}'.format(w))
    print ('Predictions')
    for x, y in zip(X_test, y_test):
        print ('true value: {}, predicted value {}'.format(y, x*w))

    loss_function = F(w, X_test, y_test)
    print ('mse: {}'.format(loss_function))
    plt.scatter(X_test, y_test)
    print_line(zip(X_test, y_test), w, 'prediction', 'red', 'solid')
    plt.show()	
