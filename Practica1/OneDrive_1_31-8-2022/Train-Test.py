import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def generate_train_test(file_name):
	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_csv(file_name, sep=',', engine='python')
	# ~ print(df)
	X = df.drop(['target'],axis=1).values   
	y = df['target'].values
	
	# ~ print ('X = {}\n y = {}'.format(X, y))
	
	#Separa el corpus cargado en el DataFrame en el 50% para entrenamiento y el 50% para pruebas
	# ~ X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle = False)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle = True)
	
	print ('X_train.shape = {}'.format(X_train.shape))
	print ('X_train = {}'.format(X_train))
	print ('y_train.shape = {}'.format(y_train.shape))
	print ('y_train = {}'.format(y_train))
	
	print ('X_test.shape = {}'.format(X_test.shape))
	print ('X_test = {}'.format(X_test))
	print ('y_test.shape = {}'.format(y_test.shape))
	print ('y_test = {}'.format(y_test))
	
	
if __name__=='__main__':
	generate_train_test('heart.csv')
	
	






















