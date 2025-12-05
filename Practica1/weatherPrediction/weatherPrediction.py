import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import sys
import pickle

#Clase para el conjunto de validación
class validation_set:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

#Clase para el conjunto de prueba
class test_set:
	def __init__(self, X_test, y_test):
		self.X_test = X_test
		self.y_test = y_test

#Clase para el conjunto de datos
class data_set:
	def __init__(self, validation_set, test_set, fold):
		self.validation_set = validation_set
		self.test_set = test_set
		self.fold = fold

#Procedimiento para el entrenamiento y pruebas de datos
def generate_train_test(file_name, kfold):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a un DataFrame
	df = pd.read_csv(file_name, sep=',', engine='python')
	X = df.drop(['RainTomorrow'],axis=1).values   
	y = df['RainTomorrow'].values
	
	#Separa el corpus cargado en el DataFrame en el 80% para entrenamiento y el 20% para pruebas
	#~ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, shuffle = False)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True)
	
	#~ print (X_train.shape)
	#~ print (X_train)
	#~ print (y_train.shape)
	#~ print (y_train)
	
	#Crea pliegues para la validación cruzada
	validation_sets = []
	kf = KFold(n_splits=kfold)
	for train_index, test_index in kf.split(X_train):
	#~ #	print("TRAIN:", train_index, "\n",  "TEST:", test_index)
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
		#~ #Agrega el pliegue creado a la lista
		validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
		

    #Almacena el conjunto de prueba
	my_test_set = test_set(X_test, y_test)

	#Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
	my_data_set = data_set(validation_sets, my_test_set, kfold) 

	return (my_data_set)

		
#Variable para los números de pliegues	
j=3		
while j != -1: 
    if __name__=='__main__':
        my_data_set = generate_train_test('weatherAUS.csv',j)
        
        print (my_data_set.test_set.X_test)
        print(type(my_data_set.test_set.X_test))
        print ('\n----------------------------------------------------------------------------------\n')
        
        #~ print (my_data_set.validation_set[0].y_train)
        
        
        
        #Guarda los pliegues en csv con el formato de nombre <num_pliegue>_<pliegue>
        i = 1
        for val_set in my_data_set.validation_set:

            #Guarda el dataset en formato csv
            np.savetxt("data_test_" + str(my_data_set.fold) + "_" + str(i) + ".csv", my_data_set.test_set.X_test, delimiter=",", fmt="%s",
            header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday")
        
            np.savetxt("target_test_" + str(my_data_set.fold) + "_" + str(i) + ".csv", my_data_set.test_set.y_test, delimiter=",", fmt="%s",
            header="RainTomorrow", comments="")

            np.savetxt("data_validation_train_" + str(my_data_set.fold) + "_" + str(i) + ".csv", val_set.X_train, delimiter=",", fmt="%s",
            header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday", comments="")
            
            #np.savetxt("data_validation_test_" + str(my_data_set.fold) + "_" + str(i) + ".csv", val_set.X_test, delimiter=",", fmt="%s",
            #header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday", comments="")
            
            np.savetxt("target_validation_train_" + str(my_data_set.fold) + "_" + str(i) + ".csv", val_set.y_train, delimiter=",", fmt="%s",
            header="RainTomorrow", comments="")
            
            #np.savetxt("target_validation_test_" + str(my_data_set.fold) + "_" + str(i) + ".csv", val_set.y_test, delimiter=",", fmt="%s",
            #header="RainTomorrow", comments="")
            i = i + 1
        
        #Guarda el dataset en pickle
        dataset_file = open ('dataset.pkl','wb')
        pickle.dump(my_data_set, dataset_file)
        dataset_file.close()
        
        dataset_file = open ('dataset.pkl','rb')
        my_data_set_pickle = pickle.load(dataset_file)
        print ("-----------------------------------------------")
        print (my_data_set_pickle.test_set.X_test)

    #Condición para cambiar los pliegues
    if j == 10:
        j = -1
    if j == 5:
        j = 10
    if j == 3:
        j = 5










