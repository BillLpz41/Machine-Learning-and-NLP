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

#Dividir el conjunto de datos de las flores
def generate_train_test_flores(file_name,fold):
	pd.options.display.max_colwidth = 200				

	#Leactura del archivo csv original
	df = pd.read_csv(file_name, sep=',', engine='python')
    #Columnas con los datos a utilizar para la predicion
	x = df.drop(['species'],axis=1).values   
    #Columna a predecir
	y = df['species'].values
	
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

#Clasficador Gaussiano para el set de validación
def class_Gaussian(data_set):
	print("\n--------Clasificacion con el metodo Gaussian--------")
	i=1
	accuracy_prom = 0
	fold_accuracy = 0
	#Obtención de las funciones gausseanas
	clf = GaussianNB()
	for val_set in data_set.validation_set:
		X= val_set.x_train	
		y= val_set.y_train
		#Entrenamiento
		clf.fit(X, y)
		#Predicción
		y_predict = clf.predict(X)
		#Exactitud de la predicción con respecto al dato de entrenamiento en este pliegue
		fold_accuracy = accuracy_score(y, y_predict)
		print ('\nAccuracy del Pliegue No.'+ str(i) +' : '+ str(fold_accuracy))
		accuracy_prom= accuracy_prom + fold_accuracy
		i=i+1
	#Promedio de exactitud
	accuracy_prom=accuracy_prom/num_Pliegues
	print('\nEl accuracy promedio para los 3 Folds es: ' + str(accuracy_prom))

#Clasificador multinomial para el set de validación	
def class_MultinomialNB(data_set):
	print("\n--------Clasificacion con el metodo Multinomial--------")
	i=1
	accuracy_prom = 0
	fold_accuracy = 0
	#Obtención de las funciones multinomiales
	clf = MultinomialNB()
	for val_set in data_set.validation_set:
		X= val_set.x_train	
		y= val_set.y_train

		#Entrenamiento
		clf.fit(X, y)
		
		#Predicción
		y_predict = clf.predict(X)
		
		#Exactitud de la predicción con respecto al dato de entrenamiento en este pliegue
		fold_accuracy = accuracy_score(y, y_predict)
		print ('\nAccuracy del Pliegue No.'+ str(i) +' : '+ str(fold_accuracy))
		accuracy_prom= accuracy_prom + fold_accuracy
		i=i+1
	#Promedio de exactitud
	accuracy_prom=accuracy_prom/num_Pliegues
	print('\nEl accuracy promedio para los 3 Folds es: ' + str(accuracy_prom))

#Predicción final con el método de mejor resultado en los emails
def mejor_Class_emails(train_set, test_set):
	print("\n--------Clasificacion de emails con el metodo Gaussian--------")
	
	#Obtención de las funciones Guasseanas
	clf = GaussianNB()
	X= train_set.x_train	
	y= train_set.y_train

	X_test = test_set.x_test
	y_test = test_set.y_test
	#Entrenamiento con todos los datos de entrenamiento 
	clf.fit(X, y)
	#Predicción con todos los datos de entrenamiento
	y_predict = clf.predict(X)
	#Obtiene los nombres de las clases
	target_names =clf.classes_
	
	print("\nClases a predecir: ")
	print (target_names)
	print("\n")
	
	#Exactitud de la predicción de los datos de entrenamiento
	accuracyM = accuracy_score(y, y_predict)
	print("\nAccuracy medido para el 70% del data set original: " + str(accuracyM))
	
	#Reporta los resultados de la clasificación de los datos de entrenamiento
	print(classification_report(y, y_predict, target_names=['0','1']))
	
	#Crea la matriz de confusión de los datos de entrenamiento
	cm = confusion_matrix(y, y_predict, labels=target_names)
	#print (cm)
	
	#Muestra la matriz de confusión en una ventana con la librería matplotlib
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
	disp.plot()
	plt.show()

	#------------------------------#
	#Predice con los datos de prueba
	y_predict = clf.predict(X_test)
	target_names =clf.classes_
	
	print("\nClases a predecir: ")
	print (target_names)
	print("\n")
	
	#Exactitud de la predicción de los datos de prueba
	accuracyM = accuracy_score(y_test, y_predict)
	print("\nAccuracy medido para el 30% del data set original: " + str(accuracyM))
	
	#Reporta los resultados de la clasificación de los datos de prueba
	print(classification_report(y_test, y_predict, target_names = ['0','1']))
	
	#Crea la matriz de confusión de los datos de prueba
	cm = confusion_matrix(y_test, y_predict, labels=target_names)
	
	#Muestra la matriz de confusión
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
	disp.plot()
	plt.show()

#Predicción final con el método de mejor resultado en las flores
def mejor_Class_flores(train_set, test_set):
	print("\n--------Clasificacion de flores con el metodo Gaussian--------")
	
	#Obtención de las funciones Guasseanas
	clf = GaussianNB()
	X= train_set.x_train	
	y= train_set.y_train

	X_test = test_set.x_test
	y_test = test_set.y_test
	#Entrenamiento con todos los datos de entrenamiento 
	clf.fit(X, y)
	#Predicción con todos los datos de entrenamiento
	y_predict = clf.predict(X)
	#Obtiene los nombres de las clases
	target_names =clf.classes_
	
	print("\nClases a predecir: ")
	print (target_names)
	print("\n")
	
	#Exactitud de la predicción de los datos de entrenamiento
	accuracyM = accuracy_score(y, y_predict)
	print("\nAccuracy medido para el 70% del data set original: " + str(accuracyM))
	
	#Reporta los resultados de la clasificación de los datos de entrenamiento
	print(classification_report(y, y_predict, target_names = target_names))
	#Crea la matriz de confusión de los datos de entrenamiento
	cm = confusion_matrix(y, y_predict, labels=target_names)
	#print (cm)

	#Muestra la matriz de confusión
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
	disp.plot()
	plt.show()

	#-------------------------------#
	#Predice con los datos de prueba
	y_predict = clf.predict(X_test)
	target_names =clf.classes_
	
	print("\nClases a predecir: ")
	print (target_names)
	print("\n")

	#Exactitud de la predicción de los datos de prueba
	accuracyM = accuracy_score(y_test, y_predict)
	print("\nAccuracy medido para el 30% del data set original: " + str(accuracyM))
	
	#Reporta los resultados de la clasificación de los datos de prueba
	print(classification_report(y_test, y_predict, target_names = target_names))
	
	#Crea la matriz de confusión de los datos de prueba
	cm = confusion_matrix(y_test, y_predict, labels=target_names)
	
	#Muestra la matriz de confusión
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
	disp.plot()
	plt.show()

if __name__=='__main__':
	#Numero de pliegues
	num_Pliegues = 3 		
	#Dividir el data set original con las flores
	print("\n---------------Clasificacion de flores---------------\n")
	my_data_set = generate_train_test_flores('iris.csv',num_Pliegues)
	
	class_Gaussian(my_data_set)
	
	class_MultinomialNB(my_data_set)
	
	mejor_Class_flores(my_data_set.train_set, my_data_set.test_set)

	print("\n---------------Clasificacion de e-mails---------------\n")
	#Dividir de nuevo el data set pero con los emails
	my_data_set = generate_train_test_email('emails.csv',num_Pliegues)
	
	class_Gaussian(my_data_set)
	
	class_MultinomialNB(my_data_set)
	
	mejor_Class_emails(my_data_set.train_set, my_data_set.test_set)
	
	print("\n---------------Fin del programa---------------\n")
	
