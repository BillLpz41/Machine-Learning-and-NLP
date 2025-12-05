from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing
from sklearn.pipeline import Pipeline

#Leactura del archivo csv original
df = pd.read_csv('mnist_train.csv', sep=',', engine='python')
#Columnas con los datos a utilizar para la predicion
x = df.drop(['label'],axis=1).values   
#Columna a predecir
y = df['label'].values

x= x/255.0
scaler = preprocessing.StandardScaler().fit(x)
X_train = scaler.transform(x)

#Creación del diccionario para definir los parámetros
#parameters = {'n_neighbors':[1,3,5,10,20,30], 'weights': ('uniform', 'distance')}
parameters = {'learning_rate_init':(0.00009, 0.00001), 'hidden_layer_sizes':[(32,16),(50,20,10)], 'alpha':(1e-4,1e-6)}

#alpha = factor de aprendizaje [1e-5, 1e-6, 1e-7, 1e-8]
#hidden_layer_sizes = numero de neuronas por capa [10, 7, 5, 2]


print("------------------------Prueba de parametros---------------------------\n")
#clf hereda las funciones del clasificador KNeighbors
clf = MLPClassifier(random_state=0, activation = 'identity', solver='sgd', batch_size=32, max_iter=300)

#GridSearch utiliza el percetron multicapa
#cv hereda las funciones del clasificador del percetron multicapa a través de GridSearch 
cv = GridSearchCV(clf, parameters, verbose=3, cv=5)

cv.fit(x, y)

#Se declaran los mejores resultados hasta el momento y el tipo de escalado
mejores_parametros_finales= cv.best_params_
mejor_puntuacion_finales = cv.best_score_

print(mejores_parametros_finales)
print(mejor_puntuacion_finales)
"""
print("------------------------Entrenamiento---------------------------\n")

clf = MLPClassifier(alpha= mejores_parametros_finales['alpha'], hidden_layer_sizes= mejores_parametros_finales['hidden_layer_sizes'], 
                    activation=mejores_parametros_finales['activation'], 
                    max_iter=mejores_parametros_finales['max_iter'], solver=mejores_parametros_finales['solver'])

clf.fit(x, y)

#Leactura del archivo csv original
df = pd.read_csv('mnist_test.csv', sep=',', engine='python')
#Columnas con los datos a utilizar para la predicion
x_test = df.drop(['label'],axis=1).values   
#Columna a predecir
y_test = df['label'].values

y_pred = clf.predict(x_test)
x_errores=[]
j=0
#Sumatoria para C+ y C-
for i in range(len(x_test)):
    if y_test[i] != y_pred[i]:
    #print("positivo")
        j=j+1
        x_errores.append(x_test[i])

 #Cálculo de la exactitud
accuracyM = accuracy_score(y_test, y_pred)
print("\nAccuracy de las pruebas {}\n".format(accuracyM))
print(j)


#Reporta los resultados de la clasificación de los datos de prueba
print(classification_report(y_test, y_pred, target_names = ['0','1','2','3', '4','5', '6', '7', '8', '9']))
#Crea la matriz de confusión de los datos de prueba
cm = confusion_matrix(y_test, y_pred)
#Muestra la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

"""