import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator
import pandas as pd
from  sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

mse_list = []
r2_list = []

exps_lineal_mse = []
exps_g2_mse = []
exps_g3_mse = []

exps_lineal_r2 = []
exps_g2_r2 = []
exps_g3_r2 = []

exps_lineal_std_mse = []
exps_g2_std_mse = []
exps_g3_std_mse = []

exps_lineal_std_r2 = []
exps_g2_std_r2 = []
exps_g3_std_r2 = []

exps_lineal_rob_mse = []
exps_g2_rob_mse = []
exps_g3_rob_mse = []

exps_lineal_rob_r2 = []
exps_g2_rob_r2 = []
exps_g3_rob_r2 = []

#Clase para el conjunto de validación
class validation_set:
	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test

#Clase para el conjunto de prueba
class test_set:
	def __init__(self, x_test, y_test):
		self.x_test = x_test
		self.y_test = y_test

#Clase para el conjunto de datos
class data_set:
    def __init__(self, validation_set, test_set, fold):
        self.validation_set = validation_set
        self.test_set = test_set
        self.fold = fold

#Procedimiento para el entrenamiento y pruebas de datos
def generate_train_test(file_name,fold):
    pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a un DataFrame
    df = pd.read_csv(file_name, sep=',', engine='python')
    #Usa todas la columnas excepto medianHouseValue
    x = df.drop(['medianHouseValue'],axis=1).values   
    #Solamente usa la columna medianHouseValue
    y = df['medianHouseValue'].values
	
	#Separa el corpus cargado en el DataFrame en el 80% para entrenamiento y el 20% para pruebas
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True, random_state=0)
	
    i=1
	#Crea pliegues para la validación cruzada
    validation_sets = []
    kf = KFold(n_splits=fold)
    for train_index, test_index in kf.split(x_train):
        #print("\nNum.Fold=",i)
        #print("TRAIN:", train_index, "\n",  "TEST:", test_index)
        X_train_, X_test_ = x_train[train_index], x_train[test_index]
        y_train_, y_test_ = y_train[train_index], y_train[test_index]
        #Agrega el pliegue creado a la lista
        validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
        i=i+1

    #Almacena el conjunto de prueba
    my_test_set = test_set(x_test, y_test)

	#Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
    my_data_set = data_set(validation_sets, my_test_set, fold) 

    return (my_data_set)

#Realiza los experimentos de acuerdo al eta0 y las iteraciones
def SGD_lineal(data_set, eta, iter):
    print("*******************") 
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train
        
        # ~ #Modelo de regresión estocástica
        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter= iter)
        regr.fit(x, y)
        y_pred = regr.predict(x)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print ('\nPliegue No.'+ str(i)+'\nmse: {} r2: {}'.format(mse, r2))
        
        i=i+1
        mse_sum= mse_sum + mse
        r2_sum= r2_sum + r2

    print("\n*******************")
            
    mse_prom=mse_sum/k
    r2_prom=r2_sum/k
    print("mse promedio: {}\nr2 promedio: {}".format(mse_prom, r2_prom))

    exps_lineal_mse.append(mse_prom)
    exps_lineal_r2.append(r2_prom)

def SGD_Grado2(data_set, eta, iter):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train
        #Conversión a polinomio de grado 2
        polynomial_features= PolynomialFeatures(degree=2)
        x_poly = polynomial_features.fit_transform(x)
        #x_poly_robust_scaler= preprocessing.RobustScaler().fit_transform(x_poly)
        """
        #Modelo lineal
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        """
        # ~ #Modelo de regresión polinomial
        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter= iter)
        regr.fit(x_poly, y)
        y_poly_pred = regr.predict(x_poly)
        mse = mean_squared_error(y, y_poly_pred)
        r2 = r2_score(y, y_poly_pred)
        
        print ('\nPliegue No.'+ str(i)+'\nmse: {} r2: {}'.format(mse, r2))
        
        i=i+1
        mse_sum= mse_sum + mse
        r2_sum= r2_sum + r2

    print("\n*******************")
            
    mse_prom=mse_sum/k
    r2_prom=r2_sum/k
    print("mse promedio: {}\nr2 promedio: {}".format(mse_prom, r2_prom))
  
    exps_g2_mse.append(mse_prom)
    exps_g2_r2.append(r2_prom)

def SGD_Grado3(data_set, eta, iter):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train
        #Conversión a polinomio de grado 2
        polynomial_features= PolynomialFeatures(degree=3)
        x_poly = polynomial_features.fit_transform(x)
        #x_poly_robust_scaler= preprocessing.RobustScaler().fit_transform(x_poly)
        """
        #Modelo lineal
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        """
        # ~ #Modelo de regresión polinomial
        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter= iter)
        regr.fit(x_poly, y)
        y_poly_pred = regr.predict(x_poly)
        mse = mean_squared_error(y, y_poly_pred)
        r2 = r2_score(y, y_poly_pred)
        
        print ('\nPliegue No.'+ str(i)+'\nmse: {} r2: {}'.format(mse, r2))
        
        i=i+1
        mse_sum= mse_sum + mse
        r2_sum= r2_sum + r2

    print("\n*******************")
            
    mse_prom=mse_sum/k
    r2_prom=r2_sum/k
    print("mse promedio: {}\nr2 promedio: {}".format(mse_prom, r2_prom))

    exps_g3_mse.append(mse_prom)
    exps_g3_r2.append(r2_prom)

#Estocástico escalado estándar
def SGD_lineal_StdScaler(data_set, eta, iter):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train
        #Conversión a polinomio lineal
        x_poly_standard_scaler = preprocessing.StandardScaler().fit_transform(x)
        #x_poly_robust_scaler= preprocessing.RobustScaler().fit_transform(x_poly)
        """
        #Modelo lineal
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        """
        # ~ #Modelo de regresión polinomial
        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter= iter)
        regr.fit(x_poly_standard_scaler, y)
        y_poly_pred = regr.predict(x_poly_standard_scaler)
        mse = mean_squared_error(y, y_poly_pred)
        r2 = r2_score(y, y_poly_pred)
        
        print ('\nPliegue No.'+ str(i)+'\nmse: {} r2: {}'.format(mse, r2))
        
        i=i+1
        mse_sum= mse_sum + mse
        r2_sum= r2_sum + r2

    print("\n*******************")
            
    mse_prom=mse_sum/k
    r2_prom=r2_sum/k
    print("mse promedio: {}\nr2 promedio: {}".format(mse_prom, r2_prom))
  
    exps_lineal_std_mse.append(mse_prom)
    exps_lineal_std_r2.append(r2_prom)

def SGD_Grado2_StdScaler(data_set, eta, iter):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train
        #Conversión a polinomio lineal
        polynomial_features= PolynomialFeatures(degree=2)
        x_poly = polynomial_features.fit_transform(x)
        x_poly_standard_scaler= preprocessing.StandardScaler().fit_transform(x_poly)
        """
        #Modelo lineal
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        """
        # ~ #Modelo de regresión polinomial
        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter= iter)
        regr.fit(x_poly_standard_scaler, y)
        y_poly_pred = regr.predict(x_poly_standard_scaler)
        mse = mean_squared_error(y, y_poly_pred)
        r2 = r2_score(y, y_poly_pred)
        
        print ('\nPliegue No.'+ str(i)+'\nmse: {} r2: {}'.format(mse, r2))
        
        i=i+1
        mse_sum= mse_sum + mse
        r2_sum= r2_sum + r2

    print("\n*******************")
            
    mse_prom=mse_sum/k
    r2_prom=r2_sum/k
    print("mse promedio: {}\nr2 promedio: {}".format(mse_prom, r2_prom))
  
    exps_g2_std_mse.append(mse_prom)
    exps_g2_std_r2.append(r2_prom)

def SGD_Grado3_StdScaler(data_set, eta, iter):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train
        #Conversión a polinomio lineal
        polynomial_features= PolynomialFeatures(degree=3)
        x_poly = polynomial_features.fit_transform(x)
        x_poly_standard_scaler= preprocessing.StandardScaler().fit_transform(x_poly)
        """
        #Modelo lineal
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        """
        # ~ #Modelo de regresión polinomial
        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter= iter)
        regr.fit(x_poly_standard_scaler, y)
        y_poly_pred = regr.predict(x_poly_standard_scaler)
        mse = mean_squared_error(y, y_poly_pred)
        r2 = r2_score(y, y_poly_pred)
        
        print ('\nPliegue No.'+ str(i)+'\nmse: {} r2: {}'.format(mse, r2))
        
        i=i+1
        mse_sum= mse_sum + mse
        r2_sum= r2_sum + r2

    print("\n*******************")
            
    mse_prom=mse_sum/k
    r2_prom=r2_sum/k
    print("mse promedio: {}\nr2 promedio: {}".format(mse_prom, r2_prom))
  
    exps_g3_std_mse.append(mse_prom)
    exps_g3_std_r2.append(r2_prom)

#Estocástico escalado robusto
def SGD_lineal_RobScaler(data_set, eta, iter):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train
        #Conversión a polinomio de grado 1
        x_poly_robust_scaler= preprocessing.RobustScaler().fit_transform(x)
        """
        #Modelo lineal
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        """
        # ~ #Modelo de regresión polinomial
        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter= iter)
        regr.fit(x_poly_robust_scaler, y)
        y_poly_pred = regr.predict(x_poly_robust_scaler)
        mse = mean_squared_error(y, y_poly_pred)
        r2 = r2_score(y, y_poly_pred)
        
        print ('\nPliegue No.'+ str(i)+'\nmse: {} r2: {}'.format(mse, r2))
        
        i=i+1
        mse_sum= mse_sum + mse
        r2_sum= r2_sum + r2

    print("\n*******************")
            
    mse_prom=mse_sum/k
    r2_prom=r2_sum/k
    print("mse promedio: {}\nr2 promedio: {}".format(mse_prom, r2_prom))
  
    exps_lineal_rob_mse.append(mse_prom)
    exps_lineal_rob_r2.append(r2_prom)

def SGD_Grado2_RobScaler(data_set, eta, iter):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train
        #Conversión a polinomio de grado 2
        polynomial_features= PolynomialFeatures(degree=2)
        x_poly = polynomial_features.fit_transform(x)
        x_poly_robust_scaler= preprocessing.RobustScaler().fit_transform(x_poly)
        """
        #Modelo lineal
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        """
        # ~ #Modelo de regresión polinomial
        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter = iter)
        regr.fit(x_poly_robust_scaler, y)
        y_poly_pred = regr.predict(x_poly_robust_scaler)
        mse = mean_squared_error(y, y_poly_pred)
        r2 = r2_score(y, y_poly_pred)
        
        print ('\nPliegue No.'+ str(i)+'\nmse: {} r2: {}'.format(mse, r2))
        
        i=i+1
        mse_sum= mse_sum + mse
        r2_sum= r2_sum + r2

    print("\n*******************")
            
    mse_prom=mse_sum/k
    r2_prom=r2_sum/k
    print("mse promedio: {}\nr2 promedio: {}".format(mse_prom, r2_prom))
  
    exps_g2_rob_mse.append(mse_prom)
    exps_g2_rob_r2.append(r2_prom)

def SGD_Grado3_RobScaler(data_set, eta, iter):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train
        #Conversión a polinomio de grado 3
        polynomial_features= PolynomialFeatures(degree=3)
        x_poly = polynomial_features.fit_transform(x)
        x_poly_robust_scaler= preprocessing.RobustScaler().fit_transform(x_poly)
        """
        #Modelo lineal
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        """
        # ~ #Modelo de regresión polinomial
        regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter = iter)
        regr.fit(x_poly_robust_scaler, y)
        y_poly_pred = regr.predict(x_poly_robust_scaler)
        mse = mean_squared_error(y, y_poly_pred)
        r2 = r2_score(y, y_poly_pred)
        
        print ('\nPliegue No.'+ str(i)+'\nmse: {} r2: {}'.format(mse, r2))
        
        i=i+1
        mse_sum= mse_sum + mse
        r2_sum= r2_sum + r2

    print("\n*******************")
            
    mse_prom=mse_sum/k
    r2_prom=r2_sum/k
    print("mse promedio: {}\nr2 promedio: {}".format(mse_prom, r2_prom))
  
    exps_g3_rob_mse.append(mse_prom)
    exps_g3_rob_r2.append(r2_prom)

if __name__=='__main__':
    #Cantidad de pliegues
    k=10    		
    my_data_set = generate_train_test('cal_housing.csv',k)
    """
    print('\nDATOS TEST:')    
    print(my_data_set.test_set.x_test)
    print('\nDATOS A PREDECIR TEST:')
    print(my_data_set.test_set.y_test)
    print ('\n----------------------------------------------------------------------------------\n')
    """
    
    #Hecho
    print("\n===== Regresión estocástico lineal sin escalar =====")
    print("\n---------------Lineal Experimento 1-----------------------------------")
    SGD_lineal(my_data_set, 0.000000001, 3000)

    print("\n---------------Lineal Experimento 2-----------------------------------")
    SGD_lineal(my_data_set, 0.000000009, 3000)

    print("\n---------------Lineal Experimento 3-----------------------------------")
    SGD_lineal(my_data_set, 0.0000000002, 3000)
    
    #PENDIENTE
    #print("\n===== Regresión polinomial estocástico grado 2 sin escalar =====")
    #print("\n---------------Grado 2 Experimento 1-----------------------------------")
    #SGD_Grado2(my_data_set, 1e-21, 1000000)

    #print("\n---------------Grado 2 Experimento 2-----------------------------------")
    #SGD_Grado2(my_data_set, 0.0000000009, 150000)

    #print("\n---------------Grado 2 Experimento 3-----------------------------------")
    #SGD_Grado2(my_data_set, 5e-11, 150000)
    """
    #PENDIENTE
    print("\n===== Regresión polinomial estocástico grado 3 sin escalar =====")
    print("\n---------------Grado 3 Experimento 1-----------------------------------")
    SGD_Grado3(my_data_set, 1e-30, 90000)

    print("\n---------------Grado 3 Experimento 2-----------------------------------")
    SGD_Grado3(my_data_set, 3e-35, 90000)

    print("\n---------------Grado 3 Experimento 3-----------------------------------")
    SGD_Grado3(my_data_set, 5e-40, 100000)
    """   
    """
    #Hecho
    print("\n===== Regresión estocástico lineal escalado estándar =====")
    print("\n---------------Lineal Experimento 1-----------------------------------")
    SGD_lineal_StdScaler(my_data_set, 0.000001, 200000)

    print("\n---------------Lineal Experimento 2-----------------------------------")
    SGD_lineal_StdScaler(my_data_set, 0.000009, 200000)

    print("\n---------------Lineal Experimento 3-----------------------------------")
    SGD_lineal_StdScaler(my_data_set, 0.0000002, 200000)
    """
    """
    #Hecho
    print("\n===== Regresión polinomial estocástico grado 2 escalado estándar =====")
    print("\n---------------Grado 2 Experimento 1-----------------------------------")
    SGD_Grado2_StdScaler(my_data_set, 0.0000008, 500000)

    print("\n---------------Grado 2 Experimento 2-----------------------------------")
    SGD_Grado2_StdScaler(my_data_set, 0.0000009, 500000)

    print("\n---------------Grado 2 Experimento 3-----------------------------------")
    SGD_Grado2_StdScaler(my_data_set, 1e-6, 500000)
    """
    """
    #Hecho
    print("\n===== Regresión polinomial estocástico grado 3 escalado estándar =====")
    print("\n---------------Grado 3 Experimento 1-----------------------------------")
    SGD_Grado3_StdScaler(my_data_set, 1e-6, 700000)

    print("\n---------------Grado 3 Experimento 2-----------------------------------")
    SGD_Grado3_StdScaler(my_data_set, 1e-5, 700000)
    
    print("\n---------------Grado 3 Experimento 3-----------------------------------")
    SGD_Grado3_StdScaler(my_data_set, 5e-6, 700000)

    """
    """
    #Hecho
    print("\n===== Regresión estocástico lineal escalado robusto =====")
    print("\n---------------Lineal Experimento 1-----------------------------------")
    SGD_lineal_RobScaler(my_data_set, 0.00001, 700000)

    print("\n---------------Lineal Experimento 2-----------------------------------")
    SGD_lineal_RobScaler(my_data_set, 0.00009, 700000)

    print("\n---------------Lineal Experimento 3-----------------------------------")
    SGD_lineal_RobScaler(my_data_set, 0.000002, 700000)
    
    #Hecho
    print("\n===== Regresión polinomial estocástico grado 2 escalado robusto =====")
    print("\n---------------Grado 2 Experimento 1-----------------------------------")
    SGD_Grado2_RobScaler(my_data_set, 0.000008, 700000)

    print("\n---------------Grado 2 Experimento 2-----------------------------------")
    SGD_Grado2_RobScaler(my_data_set, 0.000009, 700000)

    print("\n---------------Grado 2 Experimento 3-----------------------------------")
    SGD_Grado2_RobScaler(my_data_set, 1e-5, 700000)
    """
    #Hecho
    #print("\n===== Regresión polinomial estocástico grado 3 escalado robusto =====")
    #print("\n---------------Grado 3 Experimento 1-----------------------------------")
    #SGD_Grado3_RobScaler(my_data_set, 0.000000009, 2000000)

    #print("\n---------------Grado 3 Experimento 2-----------------------------------")
    #SGD_Grado3_RobScaler(my_data_set, 0.000000003, 3000000)
    
    #print("\n---------------Grado 3 Experimento 3-----------------------------------")
    #SGD_Grado3_RobScaler(my_data_set, 0.000000005, 4000000)
    

    print("\n\nResumen:")
    print("\nResultados SGD lineal:\n\tmse: {}\n\tr2: {}".format(exps_lineal_mse, exps_lineal_r2))
    print("\nResultados SGD Grado 2:\n\tmse: {}\n\tr2: {}".format(exps_g2_mse, exps_g2_r2))
    print("\nResultados SGD Grado 3:\n\tmse: {}\n\tr2: {}".format(exps_g3_mse, exps_g3_r2))

    print("\nResultados SGD lineal escalado estándar:\n\tmse: {}\n\tr2: {}".format(exps_lineal_std_mse, exps_lineal_std_r2))
    print("\nResultados SGD Grado 2:\n\tmse: {}\n\tr2: {}".format(exps_g2_std_mse, exps_g2_std_r2))
    print("\nResultados SGD Grado 3:\n\tmse: {}\n\tr2: {}".format(exps_g3_std_mse, exps_g3_std_r2))

    print("\nResultados SGD lineal escalado robusto:\n\tmse: {}\n\tr2: {}".format(exps_lineal_rob_mse, exps_lineal_rob_r2))
    print("\nResultados SGD Grado 2:\n\tmse: {}\n\tr2: {}".format(exps_g2_rob_mse, exps_g2_rob_r2))
    print("\nResultados SGD Grado 3:\n\tmse: {}\n\tr2: {}".format(exps_g3_rob_mse, exps_g3_rob_r2))

    