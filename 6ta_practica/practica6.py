from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing
from sklearn.pipeline import Pipeline

#Leactura del archivo csv original
df = pd.read_csv('heart.csv', sep=',', engine='python')
#Columnas con los datos a utilizar para la predicion
x = df.drop(['output'],axis=1).values   
#Columna a predecir
y = df['output'].values
	
#Dividr 70% de los datos para entrenamiento y 30% para pruebas
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle = True, random_state=0)

#Creación del diccionario para definir los parámetros
parameters = {'n_neighbors':[1,3,5,10,20,30], 'weights': ('uniform', 'distance')}

#Entrenamiento sin escalar

print("------------------------Entrenamiento sin escalar---------------------------\n")

#Entrenamiento con 5 pliegues (por defecto)

#clf hereda las funciones del clasificador KNeighbors
clf = KNeighborsClassifier()

#GridSearch utiliza el clasificador de vecinos cercanos
#cv hereda las funciones del clasificador de vecinos cercanos a través de GridSearch 
cv = GridSearchCV(clf, parameters, verbose=3)

cv.fit(x_train, y_train)

#Imprime el csv de los resultados
# ~ print (cv.cv_results_)
df = pd.DataFrame(cv.cv_results_)
print (df)
df.to_csv('cv_results.csv')

#Se declaran los mejores resultados hasta el momento y el tipo de escalado
mejores_parametros_finales= cv.best_params_
mejor_puntuacion_finales = cv.best_score_
mejor_escalado = 'Sin escalar'

print ('Mejores resultados sin escalar:')
print ('\tMejores parámetros: ', mejores_parametros_finales)
print ('\tMejor puntuación: ', mejor_puntuacion_finales)

#################################################
#					PIPELINES                   #
#################################################

#Entrenamiento escalado con 5 pliegues (por defecto)

#Arreglos con los tipos de escalado
scaler_names = ['Escalado estándar', 'Escalado robusto']
scalers = [preprocessing.StandardScaler(), preprocessing.RobustScaler()]
clf = KNeighborsClassifier()

#Creación del diccionario con los parámetros
parameters_pipeline = [{'clf__n_neighbors':[1,3,5,10,20,30], 'clf__weights': ('uniform', 'distance')}]

#Entrenamiento con cada tipo de escalado
for scaler_name, scaler in zip(scaler_names, scalers):
	print("--------------------------------------------------\n")
	print ('Método de escalado: ', scaler_name, '\n')
	
	#Usa la función Pipeline para pasarlo al GridSearchCV
	pipeline = Pipeline([
					('scalers', scaler),
					('clf', clf)
					])

	#Entrenamiento con los diferentes escalados y el mismo clasificador		
	cv = GridSearchCV(pipeline, param_grid=parameters_pipeline, verbose=3)
	cv.fit(x_train, y_train)
	
	print ('\nMejores resultados: ')
	print ('\tMejor puntuación: ', cv.best_params_)
	print ('\tMejores parámetros: ', cv.best_score_)
	
	#Compara el resultado actual con el nuevo y escoge mejor 
	if (mejor_puntuacion_finales < cv.best_score_):
		mejor_escalado = scaler_name
		mejor_puntuacion_finales = cv.best_score_
		mejores_parametros_finales = cv.best_params_

print("\n----------------------Mejores resultados globales-----------------------------")
print ('\nMejor escalado: ', mejor_escalado)
print ('\nMejores parámetros: ', mejores_parametros_finales)
print ('\nMejor puntuación: ', mejor_puntuacion_finales)


print("\n-----------------------Train/Test final----------------------------")


#Entrenamiento con la función de clasificación de forma directa
# - GridSearchCV no servirá debido a que lo hace en pliegues
clf = KNeighborsClassifier(n_neighbors= mejores_parametros_finales['clf__n_neighbors'], weights= mejores_parametros_finales['clf__weights'])
#Entrenamiento con el 70%
clf.fit(x_train, y_train)

#Prueba
y_pred = clf.predict(x_test)

#Evaluación del modelo
print ('\nSe obtuvo un accuracy final de: ', accuracy_score(y_test,y_pred))
print ('\nReporte de clasificación: \n', classification_report(y_test, y_pred))

#Muestra la matriz de confusión
cm = confusion_matrix(y_test,y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print("-------------------------Fin del programa--------------------------")

