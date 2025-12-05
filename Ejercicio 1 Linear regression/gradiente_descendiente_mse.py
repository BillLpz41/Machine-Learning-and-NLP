import matplotlib.pyplot as plt
import sys

def F(w, X, y):
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
	X = [1, 2, 3, 4, 5, 6]
	y = [1, 2.5, 2, 4, 4.5, 6.3]
	
	iterations = int(sys.argv[1])
	plt.scatter(X, y)
	
	w= 0
	alpha = 0.01
	# ~ alpha = 0.05 #Efecto similar al de no sacar el promedio
	for t in range(iterations):
		loss_function = F(w, X, y)
		gradient = dF(w, X, y)
		w = w - alpha * gradient
		print ('iteration {}: w = {}, F(w) = {}'.format(t, w, loss_function))
		print_line(zip(X, y), w, t)

	print_line(zip(X, y), w, t, 'red', 'solid')
	plt.show()

	
		
