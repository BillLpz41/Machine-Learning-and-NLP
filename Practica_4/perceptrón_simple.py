import numpy as np
from sklearn.metrics import accuracy_score


def activation_function (predicted_values):
	threshold_values = []
	
	for value in predicted_values:
		if value <0:
			threshold_values.append(0)
		else:
			threshold_values.append(1)
	
	return (threshold_values)
	
def weight_adjustment(y_predicted, y_train, weights, x_train):
	for i in range(len(y_train)):
		print ('y_train: {} - y_predicted: {}'.format(y_train[i], y_predicted[i]))
		error = y_train[i] - y_predicted[i]
		print ('error: ', error)
		if error != 0:
			weights += np.sum([weights,np.multiply(x_train[i], error)], axis=0)
			print ('weights: {}'.format(weights)) 
	return (weights)
	
	
if __name__ == "__main__":
	x_train = np.array([[1, 0, 1], 
						[1, 1, 0], 
						[0, 1, 0]])
	
	y_train = [0, 1, 1]
	weights = np.array([0, 0, 0]) #-->[ 0
	                                   #0
	                                   #0
	                                  #] 
	# ~ weights = np.array([0.3, 0.1, 0])
	
	
	epochs = 5
	for i in range (epochs):
		print ('----------------Iteración ', i, ' -------------------\n')
		weight_sums = np.dot(x_train,weights.T)# Hacemos la transpuesta, aunque al ser los pesos un vector de una dimensión python no requiere hacer la transpuesta para resolver la multiplicación porque lo hace mediante el producto punto u.v = u1.v1 + u2.v2 + ... + un.vn
		print ('weight_sums:\n', weight_sums)
		y_predicted = activation_function(weight_sums)
		print ('y_predicted:', y_predicted)
		print ('y_true: ', y_train)
		
		print ('accuracy: ', accuracy_score(y_train, y_predicted))
		
		weights = weight_adjustment(y_predicted, y_train, weights, x_train)
	
	print ('final weights :', weights)
	print ('final accuracy: ', accuracy_score(y_train, y_predicted))
