from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from  sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import os

print('Cargando dataset...\n')

# Leemos el conjunto de datos
df = pd.read_csv('mnist_train.csv', sep=',', engine='python')
# Separamos las columnas con los datos de entrada y la columna con la etiqueta
X = df.drop(['label'], axis=1).values
y = df['label'].values

# Preprocesando los datos de entrada
print('Escalando los datos...\n')
scaler = preprocessing.StandardScaler().fit(X)
X_train = scaler.transform(X)
"""
#Buscando los mejores parámetros
parameters={'hidden_layer_sizes':[(10,9),(32,16),(50,20,10),(100),(300,150,),(400,200,),(300,150,75)],
            'learning_rate_init':(0.1,0.001,0.00001), 
            'alpha':(0.1,0.2,0.01,1e-4,1e-6),
            'activation':('relu','identity'),
            'solver':('adam','sgd'),
            'max_iter':(50,100,200)}

# Creamos un perceptrón multicapa
clf = MLPClassifier(random_state=0)

cv= GridSearchCV(clf, parameters, verbose=3, cv=5)

# Entrenando el modelo
cv.fit(X_train, y)

# Predicciones sobre el conjunto de prueba
y_pred = cv.predict(X)

# Evaluando el rendimiento del modelo
accuracy = accuracy_score(y, y_pred)
print("Exactitud: ", accuracy)

# Mostramos el reporte de clasificación
print(classification_report(y, y_pred))

mejores_parametros_finales= cv.best_params_
mejor_puntuacion_finales = cv.best_score_

print('Mejores parámetros finales:', mejores_parametros_finales)
print('Mejor puntuación final:', mejor_puntuacion_finales)
"""

print("------------------------Entrenando con todo el conjunto con los mejores parámetros---------------------------\n")
print("random_state=0, alpha= 0.2, hidden_layer_sizes= (300,150,75), activation='relu', max_iter=100, solver='adam'\n")

clf = MLPClassifier(random_state=0, alpha= 0.2, hidden_layer_sizes= (300,150,75), activation='relu', max_iter=100, solver='adam')
# Entrenando el modelo
clf.fit(X, y)

# Predicciones sobre el conjunto de prueba
y_pred = clf.predict(X)

# Evaluando el rendimiento del modelo
accuracy = accuracy_score(y, y_pred)
print("Accuracy de entrenamiento: ", accuracy)

# Muestra el reporte de clasificación
print(classification_report(y, y_pred))

print("------------------------Uso del conjunto de pruebas---------------------------\n")
print('Cargando conjunto de prueba...\n')
#Leactura del csv de prueba
df = pd.read_csv('mnist_test.csv', sep=',', engine='python')
#Columnas con los datos a utilizar para la predicion
x_test = df.drop(['label'],axis=1).values   
#Columna a predecir
y_test = df['label'].values

# Usa los mismos parámetros del clasificador para el conjunto de prueba
y_pred = clf.predict(x_test)

#Cálculo de la exactitud
accuracyM = accuracy_score(y_test, y_pred)
print("\nAccuracy de las pruebas: ", accuracyM)

#Reporta los resultados de la clasificación de los datos de prueba
print(classification_report(y_test, y_pred, target_names = ['0','1','2','3', '4','5', '6', '7', '8', '9']))
#Crea la matriz de confusión de los datos de prueba
cm = confusion_matrix(y_test, y_pred)
#Muestra la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Arreglo para contener los dígitos mal predichos
x_errores=[]
y_errores=[]
y_correctos=[]

cant=0

# Recolecta los dígitos mal predichos y los cuenta
# i recorre el tamaño de x_test, ya que la cantidad de instancias es la misma que en y_test
for i in range(len(x_test)):
    # Compara si hizo correctamente la predicción
    if y_test[i] != y_pred[i]:
        # Cuenta los errores
        cant=cant+1
        # Añade el error
        x_errores.append(x_test[i])
        y_errores.append(y_pred[i])
        y_correctos.append(y_test[i])

print('Cantidad de errores: ', cant)

#Crea un archivo csv para facilitar la comparación de resultados mal predichos
datos=[[],[]]
        
datos[0]= y_correctos
datos[1]= y_errores
datos=np.column_stack([datos[0], datos[1]])    

np.savetxt("predicciones_fallidas.csv", datos, delimiter=",", fmt="%s", header="Prediction, Resultado", comments="")

# Imprime de forma gráfica los números que fueron mal predichos 
i=0 #<- Contador de imágenes
# Crea un directorio de nombre Autoimages para guardar las imágenes y previene algún error si este ya existe
print("Guardando gráficas de las predicciones fallidas...")
os.makedirs('Autoimages', exist_ok=True)

for img in x_errores:
    # Ajusta la gráfica y lo pone en escala de grises
    image = np.reshape(img, (28, 28))
    plt.imshow(image, cmap="Greys")
    plt.title('Real: '+ str(y_correctos[i]) + '; Predicho: '+ str(y_errores[i]), fontsize=30)
    #-> Guarda las gráficas en una carpeta llamada Autoimages (la carpeta DEBE estar creada con anterioridad)
    #-> Si quieres guardar las gráficas, descomenta las siguientes dos líneas de código y la línea de código anterior al for
    plt.savefig('./Autoimages/'+str(i)+'.png')
    i=i+1
    
    #-> Muestra las gráficas cuya predicción fue fallida
    #-> Si quieres ver las gráficas, descomenta la siguiente línea de código   
    #plt.show()
