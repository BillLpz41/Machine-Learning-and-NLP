import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
	def __init__(self, validation_set, test_set, train_set):
		self.validation_set = validation_set
		self.test_set = test_set
		self.train_set = train_set

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
	
	#Crear pliegues para la validación cruzada
	i=1
	validation_sets = []
	kf = KFold(n_splits=10)
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
	#my_data_set = final_dset(my_test_set, my_train_set) 
	
	my_data_set = data_set(validation_sets, my_test_set, my_train_set)
	
	
	
	return (my_data_set)

def hiperplano_fold(data_set):
	k=1
	sum_prom=0
	for val_set in data_set.validation_set:
		#y_pred=[]
		print("========= PLIEGUE ",k, "==========")
		x_train= val_set.x_train
		y_train= val_set.y_train

		#print(x_train)
		#print(y_train)
		x_test= val_set.x_test
		y_test= val_set.y_test

		#print(x_train[0])
		#Cálculo de C+ y C-
		c_pos=0
		c_neg=0

		#Suma de los puntos positivos y negativos
		sum_points_pos=0
		sum_points_neg=0

		#Cantidad de positivos y negativos
		M_pos=0
		M_neg=0
		#print(len(x_train))
		#Sumatoria para C+ y C-
		for i in range(len(x_train)):
			if y_train[i] == 1:
				#print("positivo")
				sum_points_pos= sum_points_pos + x_train[i]
				#print('\n',x_train[i])
				M_pos= M_pos + 1
			#if y_train[i] == 0:
			else:
				#print("negativo")
				sum_points_neg= sum_points_neg + x_train[i]
				#print('\n',x_train[i])
				M_neg= M_neg + 1

		#print(M_pos)
		c_pos= sum_points_pos/M_pos
		print('Promedio de positivos: ', c_pos)

		"""
		#Cálculo de C-
		
		for i in range(len(x_train)):
			if y_train[i] == 0:
				sum_points_neg= sum_points_neg + x_train[i]
		"""
		#print(M_neg)
		c_neg= sum_points_neg/M_neg
		print('\nPromedio de negativos: ', c_neg)
		
		#Cálculo del punto intermedio C
		c= (c_pos+c_neg)/2
		print('\nPunto intermedio C: ', c)
		
		#Calculando la magnitud
		suma_cuadrada=0
		for eje in c:
			#print(eje)
			suma_cuadrada= suma_cuadrada + (eje**2)
			#print(suma_cuadrada)
		
		cmag= math.sqrt(suma_cuadrada)
		print('Magnitud de C: ', cmag)
		y_pred=[]
		#Cálculo de las proyecciones de los datos de prueba
		for point in x_test:
			#print(point)
			proyeccion(point, c, cmag, y_pred)

		"""
		#Cálculo de producto punto
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
		#Cálculo del accuracy
		accuracyM = accuracy_score(y_test, y_pred)
		sum_prom= sum_prom + accuracyM
		print("\nAccuracy medido en el pliegue {}: {}\n".format(k, accuracyM))
		#y_pred=[]
		k=k+1

	sum_prom=sum_prom/k
	print("Accuracy promedio: ", sum_prom)


   
def proyeccion(p_test, p_intermedio, magnitud, pred):
  
	#print('antes',p_test)
	#p_test= p_test.flatten
	#print('despues',p_test)   
	#p_intermedio= p_intermedio[:, np.newaxis]

	prod_pt= np.dot(p_test, p_intermedio)
	proy=prod_pt/magnitud
	print("Dato ", p_test,"\n")
	#print("La proyección es de: ", proy/magnitud)
	if proy <= magnitud:
		#Pertenece a la clase negativa
		#print("\nEl dato", p_test,"es 0. \n\tProyeccion de {}\n".format(proy))
		pred.append(0)
	else:
		#Pertenece a la clase positiva
		#print("\nEl dato", p_test,"es 1. \n\tProyeccion de {}\n".format(proy))
		pred.append(1)
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

if __name__=='__main__':
	
	print("\n---------------Hiperplano con heart---------------\n")
	#Dividir de nuevo el data set pero con los emails
	#my_data_set = generate_train_test_email('emails.csv',num_Pliegues)
	
	my_dset= generate_dset('heart_mini.csv')
	hiperplano_fold(my_dset)
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

