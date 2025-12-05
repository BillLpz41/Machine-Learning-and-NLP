import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from  sklearn import preprocessing
import matplotlib.pyplot as plt

class validation_set:
	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test

class train_set:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

class test_set:
	def __init__(self, x_test, y_test):
		self.x_test = x_test
		self.y_test = y_test


class data_set:
	def __init__(self, validation_set, test_set, train_set, fold):
		self.validation_set = validation_set
		self.test_set = test_set
		self.train_set = train_set
		self.fold = fold


def generate_train_test(file_name,fold):
	pd.options.display.max_colwidth = 200				

	#Leactura del archivo csv original
	df = pd.read_csv(file_name, sep=',', engine='python')
	#Conversión de todo el dataframe a flotante (para evitar conflictos futuros)
	df = df.astype(float)
    #Columnas con los datos a utilizar para la predicion
	x = df.drop(['target'],axis=1).values   
    #Columna a predecir
	y = df['target'].values
	
	#Dividr 70% de los datos para entrenamiento y 30% para pruebas
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle = True, random_state=0)
	
	#Crear pliegues para la validación cruzada
	i=1
	validation_sets = []
	kf = KFold(n_splits=fold)
	for train_index, test_index in kf.split(x_train):
    
		X_train_, X_test_ = x_train[train_index], x_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
       
		validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
		i=i+1    

	my_test_set = test_set(x_test, y_test)

	my_train_set = train_set(x_train, y_train)

	my_data_set = data_set(validation_sets, my_test_set, my_train_set, fold) 

	return (my_data_set)
	
def activation_function (predicted_values):
	threshold_values = []
	
	for value in predicted_values:
		if value <0.0:
			threshold_values.append(0)
		else:
			threshold_values.append(1)
	
	return (threshold_values)
	
def weight_adjustment(y_predicted, y_train, weights, x_train):
	for i in range(len(y_train)):
		#print ('y_train: {} - y_predicted: {}'.format(y_train[i], y_predicted[i]))
		error = y_train[i] - y_predicted[i]
		#print (i,'.-error: ', error)
		if error != 0:
			weights += np.sum([weights,np.multiply(x_train[i], error)], axis=0)
		#	print ('weights: {}'.format(weights)) 
	return (weights)
	
def perceptron_S_val(data_set, epochs, num_Pliegues):
	print('\n----------------------Prueba con ', epochs,' epocas-------------------------')
	N_pliegue=1
	i=1
	
	accuracy_prom = 0
	fold_accuracy = 0
	
	for val_set in data_set.validation_set:
		print('\n--------------------------Pliegue ', N_pliegue,'-------------------------')

		x_train= val_set.x_train	
		y_train= val_set.y_train
		
		weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		#Conversión del arreglo entero a flotante para evitar conflictos de operaciones
		weights = weights.astype(float)
		
		
	
		
		for i in range (epochs):
			print ('\n------> Iteración(epoca) ', i, '	\n')
			weight_sums = np.dot(x_train,weights.T)
			
			
			#Escalado------------
			weight_sums = np.reshape(weight_sums, (-1,1))
			weight_sums= preprocessing.StandardScaler().fit_transform(weight_sums)
			weight_sums = np.reshape(weight_sums, (-1,))
			#print ('weight_sums:\n', weight_sums)
			#--------------------
			
			y_predicted = activation_function(weight_sums)
			#print('y_predicted:',y_predicted)
			#print('y_real:',y_train)
			print ('Accuracy: ', accuracy_score(y_train, y_predicted))
			weights = weight_adjustment(y_predicted, y_train, weights, x_train)
			
						
			weights = np.reshape(weights, (-1,1))
			weights= preprocessing.StandardScaler().fit_transform(weights)
			weights = np.reshape(weights, (-1,))
			
			
			#print ('weights:', weights)
		
			fold_accuracy = accuracy_score(y_train, y_predicted)
			
		print ('final weights :', weights)
		print ('final accuracy: ', fold_accuracy)
		accuracy_prom= accuracy_prom + fold_accuracy
		N_pliegue=N_pliegue+1
		
	accuracy_prom=accuracy_prom/num_Pliegues
	print('\nEl accuracy promedio para los 3 Folds es: ' + str(accuracy_prom))

def perceptron_S_train(train_set, epochs):
	print('\n----------------------Prueba train_set con', epochs,' epocas-------------------------')
	N_pliegue=1
	i=1
	
	accuracy_prom = 0
	fold_accuracy = 0
	
	x_train= train_set.x_train	
	y_train= train_set.y_train	
	
	weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	#Conversión del arreglo entero a flotante para evitar conflictos de operaciones
	weights = weights.astype(float)
			
	for i in range (epochs):
		print ('\n------> Iteración(epoca) ', i, '	\n')
		weight_sums = np.dot(x_train,weights.T)
		
		
		#Escalado------------
		weight_sums = np.reshape(weight_sums, (-1,1))
		weight_sums= preprocessing.StandardScaler().fit_transform(weight_sums)
		weight_sums = np.reshape(weight_sums, (-1,))
		#print ('weight_sums:\n', weight_sums)
		#--------------------
		
		y_predicted = activation_function(weight_sums)
		#print('y_predicted:',y_predicted)
		#print('y_real:',y_train)
		print ('Accuracy: ', accuracy_score(y_train, y_predicted))
		weights = weight_adjustment(y_predicted, y_train, weights, x_train)
		
					
		weights = np.reshape(weights, (-1,1))
		weights= preprocessing.StandardScaler().fit_transform(weights)
		weights = np.reshape(weights, (-1,))
		
		
		#print ('weights:', weights)
	
		fold_accuracy = accuracy_score(y_train, y_predicted)
		
	print ('final weights :', weights)
	print ('final accuracy: ', fold_accuracy)

def perceptron_S_test(test_set, epochs):
	print('\n----------------------Prueba test_set con', epochs,' epocas-------------------------')
	N_pliegue=1
	i=1
	
	accuracy_prom = 0
	fold_accuracy = 0
	
	x= test_set.x_test
	y= test_set.y_test	
	
	weights = np.array([-1.08462998, 0.6434383, 0.74973116, -2.11370193, -1.48681049, 0.67596869, 0.63777837, -1.15254514, 0.61567311, 0.54530094, 0.70529623, 0.67550161, 0.58899913])
	
	#Conversión del arreglo entero a flotante para evitar conflictos de operaciones
	weights = weights.astype(float)
			
	for i in range (epochs):
		print ('\n------> Iteración(epoca) ', i, '	\n')
		weight_sums = np.dot(x,weights.T)
		
		
		#Escalado------------
		weight_sums = np.reshape(weight_sums, (-1,1))
		weight_sums= preprocessing.StandardScaler().fit_transform(weight_sums)
		weight_sums = np.reshape(weight_sums, (-1,))
		#print ('weight_sums:\n', weight_sums)
		#--------------------
		
		y_predicted = activation_function(weight_sums)
		y_real = y.astype(int)
		print('y_predicted:',y_predicted)
		print('y_real:',y_real)
		print ('Accuracy: ', accuracy_score(y, y_predicted))
		weights = weight_adjustment(y_predicted, y, weights, x)
		
					
		weights = np.reshape(weights, (-1,1))
		weights= preprocessing.StandardScaler().fit_transform(weights)
		weights = np.reshape(weights, (-1,))
		#print ('weights:', weights)
	
		fold_accuracy = accuracy_score(y, y_predicted)
		
	print ('final weights :', weights)
	print ('final accuracy: ', fold_accuracy)
	

	print (classification_report(y_real, y_predicted, target_names=['0.0','1.0']))
	
	cm = confusion_matrix(y_real,y_predicted)
	print (cm)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot()
	plt.show()

	datos=[[],[]]
    
    #y_test=y_test[:, np.newaxis]
    #print(y_test)
    #y_poly_pred= y_pred[:, np.newaxis]
    
	datos[0]= y_real
	datos[1]= y_predicted
	datos=np.column_stack([datos[0], datos[1]])    

    #print("dato: {}".format(datos))

	np.savetxt("prediccion.csv", datos, delimiter=",", fmt="%s", header="Real, Prediccion", comments="")
	
if __name__ == "__main__":
	
	#Numero de pliegues
	num_Pliegues = 3 
	
	epochs = 4
	"""	
	#Dividir el data set original
	my_data_set = generate_train_test('heart.csv',num_Pliegues)
	
	perceptron_S_val(my_data_set, epochs, num_Pliegues)
	
	
	print("\n---------------Fin del los pliegues con", epochs,"epocas -------------\n")

	#Numero de pliegues
	num_Pliegues = 3 	
	epochs = 9
	#Dividir el data set original
	my_data_set = generate_train_test('heart.csv',num_Pliegues)
	
	perceptron_S_val(my_data_set, epochs, num_Pliegues)
	
	
	print("\n---------------Fin del los pliegues con", epochs,"epocas -------------\n")

	#Numero de pliegues
	num_Pliegues = 3 	
	epochs = 10
	"""
	#Dividir el data set original
	my_data_set = generate_train_test('heart.csv',num_Pliegues)
	
	perceptron_S_val(my_data_set, epochs, num_Pliegues)
	
	
	print("\n---------------Fin del los pliegues con", epochs,"epocas -------------\n")
	print("\n---------------Fin del los pliegues-------------\n")
	

	best_epochs = 3

	perceptron_S_train(my_data_set.train_set, best_epochs)

	print("\n---------------Fin del train_set-------------\n")

	perceptron_S_test(my_data_set.test_set, best_epochs)

	print("\n---------------Fin del test_set-------------\n")


	
