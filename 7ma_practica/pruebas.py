from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing
from sklearn.pipeline import Pipeline
import numpy as np
import os

#Leactura del archivo csv original
df = pd.read_csv('mnist_test.csv', sep=',', engine='python')
#Columnas con los datos a utilizar para la predicion
x_test = df.drop(['label'],axis=1).values   
#Columna a predecir
y_test = df['label'].values

x_errores = []
j=0
#Sumatoria para C+ y C-
for i in range(20):
	x_errores.append(x_test[i])

os.makedirs('Pruebas', exist_ok=True)

i=0
for img in x_errores:
	image = np.reshape(img, (28, 28))
	plt.imshow(image, cmap="Greys")
	plt.title('Real', fontsize=30)
	plt.savefig('./Pruebas/'+str(i)+'.png')
	i=i+1
	#plt.show()
