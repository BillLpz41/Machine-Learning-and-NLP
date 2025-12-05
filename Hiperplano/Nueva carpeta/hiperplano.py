import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import math

y_pred=[]

#Clase para el conjunto de validación
class validation_set:
	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test

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

#Clase para el conjunto de datos
class data_set:
	def __init__(self, validation_set, test_set, train_set, fold):
		self.validation_set = validation_set
		self.test_set = test_set
		self.train_set = train_set
		self.fold = fold
		
class final_dset:
    def __init__(self, test_set, train_set):
         self.test_set = test_set
         self.train_set = train_set
"""
#Dividir el conjunto de datos de los emails
def generate_train_test_email(file_name,fold):
	pd.options.display.max_colwidth = 200				

	#Leactura del archivo csv original
	df = pd.read_csv(file_name, sep=',', engine='python')
    #Columnas con los datos a utilizar para la predicion
	x = df.drop(['Prediction', 'Email No.'],axis=1).values   
    #Columna a predecir
	y = df['Prediction'].values
	
	#Dividr 70% de los datos para entrenamiento y 30% para pruebas
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle = True, random_state=0)
	
	#Crear pliegues para la validación cruzada
	i=1
	validation_sets = []
	kf = KFold(n_splits=fold)
	for train_index, test_index in kf.split(x_train):
    
		X_train_, X_test_ = x_train[train_index], x_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
        #Añade los conjuntos de validación
		validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
		i=i+1    
	#Crea un test set
	my_test_set = test_set(x_test, y_test)
	#Crea un train set
	my_train_set = train_set(x_train, y_train)
	#Crea un data set con el validation set, test set y train set
	my_data_set = data_set(validation_sets, my_test_set, my_train_set, fold) 

	return (my_data_set)
"""

   

def hiperplano_emails(file_name):
    pd.options.display.max_colwidth = 200				

	#Leactura del archivo csv original
    df = pd.read_csv(file_name, sep=',', engine='python')
    #Columnas con los datos a utilizar para la predicion
    x = df.drop(['Prediction', 'Email No.'],axis=1).values   
    #Columna a predecir
    y = df['Prediction'].values
	
	#Dividr 70% de los datos para entrenamiento y 30% para pruebas
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle = True, random_state=0)

    #Crea un test set
	#my_test_set = test_set(x_test, y_test)
	#Crea un train set
	#my_train_set = train_set(x_train, y_train)
	#Crea un data set con el validation set, test set y train set
	#my_data_set = data_set(validation_sets, my_test_set, my_train_set, fold)
	#my_data_set= final_dset(my_test_set, my_train_set)
	
	#print(x_train[0])
    c_1=0
	#l= len(x_train)
	#print(l)
    sum_points_1=0
    for i in range(len(x_train)):
        if y_train[i] == 1:
            sum_points_1= sum_points_1 + x_train[i]

    c_1= (1/(len(x_train)))*(sum_points_1)
    print('Promedio de 1s',c_1 )
	
    c_0=0
    l= len(x_train)
    print(l)
    sum_points_0=0
    for i in range(len(x_train)):
        if y_train[i] == 0:
            sum_points_0= sum_points_0 + x_train[i]

    c_0= (1/(len(x_train)))*(sum_points_0)
    print('Promedio de 0s',c_0 )
	
    c= (c_1+c_0)/2
    print('Punto intermedio C: ', c)
    suma_cuadrada=0
	
    for i in c:
        suma_cuadrada= suma_cuadrada + (c[1]*c[1])
	
    cmag= math.sqrt(suma_cuadrada)
    print('Magnitud de C: ', cmag)

    for point in x_test:
        #print(point)
        proyeccion(point, c, cmag)

    """
    prod_punto=0
    proy=0
    for j in x_test:
        prod_punto= prod_punto + (x_test[j]*c[j])
        print(prod_punto)
        proy= prod_punto/cmag
        if proy < cmag:
            y_pred.append(0)
        else:
            y_pred.append(1)
    """

    accuracyM = accuracy_score(y_test, y_pred)
    print("\nAccuracy medido para el 30% del data set original: " + str(accuracyM))
    
    #print(y_pred)

    datos=[[],[]]
    
    #y_test=y_test[:, np.newaxis]
    #print(y_test)
    #y_poly_pred= y_pred[:, np.newaxis]
    
    datos[0]= y_test
    datos[1]= y_pred
    datos=np.column_stack([datos[0], datos[1]])    

    #print("dato: {}".format(datos))

    np.savetxt("prediccion.csv", datos, delimiter=",", fmt="%s", header="Prediction, hiperplane", comments="")
        
def proyeccion(p_test, p_intermedio, magnitud):
    """
    prod_punto=0

    for j in range(len(p_test)):
        prod_punto= prod_punto + (p_test[j]*p_intermedio[j])
        print(prod_punto)
        proy= prod_punto/magnitud
        if proy < magnitud:
            y_pred.append('0')
        else:
            y_pred.append('1')
    """
    #print('antes',p_test)
    #p_test= p_test.flatten
    #print('despues',p_test)   
    #p_intermedio= p_intermedio[:, np.newaxis]
    proy= np.dot(p_test, p_intermedio)/magnitud
    #print("La proyección es de: ", proy/magnitud)
    if proy < magnitud:
        #print("El dato" + p_test + "es 0")
        y_pred.append(0)
    else:
        #print("El dato" + p_test + "es 1")
        y_pred.append(1)
    	
if __name__=='__main__':
	"""
	#Numero de pliegues
	num_Pliegues = 3 		
	#Dividir el data set original con las flores
	print("\n---------------Hiperplano con flores Iris---------------\n")
	my_data_set = generate_train_test_flores('iris.csv',num_Pliegues)
	
	hiperplano('iris.csv', 'species')
	
	hiperplano_test(my_data_set.train_set, my_data_set.test_set)
    """
	print("\n---------------Hiperplano con e-mails---------------\n")
	#Dividir de nuevo el data set pero con los emails
	#my_data_set = generate_train_test_email('emails.csv',num_Pliegues)
	
	hiperplano_emails('emails.csv')
	
	#hiperplano_test(my_data_set.train_set, my_data_set.test_set)
	
	print("\n---------------Fin del programa---------------\n")
	



"""

#positive = np.array([5,7], [6,6], [5,5], [4,5], [4,6])
#negative = np.array([1,1], [2,1], [3,1], [1,2], [2,2])


Xpos = np.array([5, 6, 5, 4, 4])
Ypos = np.array([7, 6, 5, 5, 6])

Xneg= np.array([1, 2, 3, 1, 2])
Yneg= np.array([1, 1, 1, 2, 2])
#Grafica los puntos
plt.scatter(Xpos,Ypos)
plt.scatter(Xneg, Yneg)

sum_Xpos=0
sum_Ypos=0

for x in Xpos:
    sum_Xpos=sum_Xpos+x
    
for y in Ypos:    
    sum_Ypos=sum_Ypos+y

cxpos=(1/(len(Xpos)))*(sum_Xpos)
cypos=(1/(len(Ypos)))*(sum_Ypos)

#plt.scatter(cxpos, cypos)

sum_Xneg=0
sum_Yneg=0

for x in Xneg:
    sum_Xneg=sum_Xneg+x
    
for y in Yneg:    
    sum_Yneg=sum_Yneg+y

cxneg=(1/(len(Xneg)))*(sum_Xneg)
cyneg=(1/(len(Yneg)))*(sum_Yneg)

plt.scatter(cxneg, cyneg, color="black")

cx= (cxpos+cxneg)/2
cy= (cypos+cyneg)/2

plt.scatter(cx, cy)

cmag= math.sqrt((cx*cx)+(cy*cy))

print('Magnitud de C: ',cmag)
#TEST
X=[3,3]
Y=[2,4]
Z=[4,4]

proyeccion(X, [cx,cy], cmag)
proyeccion(Y, [cx,cy], cmag)
proyeccion(Z, [cx,cy], cmag)



plt.show()

"""

