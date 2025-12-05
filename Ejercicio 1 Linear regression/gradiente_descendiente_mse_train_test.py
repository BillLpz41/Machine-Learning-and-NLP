import matplotlib.pyplot as plt
import sys

def F(w, X, y):
	a=sum((w * x - y)**2 for x, y in zip(X, y))
	print('a {}'.format(a))
	return sum((w * x - y)**2 for x, y in zip(X, y))/len(y)


def dF(w, X, y):
	return sum(2*(w * x - y) * x for x, y in zip(X, y))/len(y)


def print_line(points, w, iteration, line_color = None, line_style = 'dotted'):
	list_x = []
	list_y = []
	for index, tuple in enumerate(points):
		x = tuple[0]
		y = x * w
		list_x.append(x)
		list_y.append(y)
	plt.text(x,y, iteration, horizontalalignment='right')
	plt.plot(list_x, list_y, color = line_color, linestyle= line_style)

if __name__=='__main__':
	
	#Train
	X_train = [1, 2, 3, 4, 5, 6]
	y_train = [1, 2.5, 2, 4, 4.5, 6.3]
	
	iterations = int(sys.argv[1])
	plt.scatter(X_train, y_train)
	
	print ('X = {}\n y = {}'.format(X_train, y_train))
	print('\n')
	print ('X = {}\n y = {}'.format(X_train[2], y_train[2]))

	w= 0
	alpha = 0.01
	# ~ alpha = 0.05 #Efecto similar al de no sacar el promedio
	for t in range(iterations):
		print('before loss function w= {}'.format(w))
		loss_function = F(w, X_train, y_train)
		gradient = dF(w, X_train, y_train)
		print('gradient {}'.format(gradient))

		w = w - alpha * gradient
		print('w= {}'.format(w))
		print ('iteration {}: w = {}, F(w) = {}'.format(t, w, loss_function))
		print_line(zip(X_train, y_train), w, t)

	print_line(zip(X_train, y_train), w, t, 'red', 'solid')
	plt.show()


	#Test
	X_test = [1.5, 3, 7]
	y_test = [1.8, 2.5, 7]
	
	print ('Calculated weight: {}'.format(w))
	print ('Predictions')
	for x, y in zip(X_test, y_test):
		print ('true value: {}, predicted value {}'.format(y, x*w))

	loss_function = F(w, X_test, y_test)
	print ('mse: {}'.format(loss_function))
	plt.scatter(X_test, y_test)
	print_line(zip(X_test, y_test), w, 'prediction', 'red', 'solid')
	plt.show()	
