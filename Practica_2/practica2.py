import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from  sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from array import *

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

class train_set:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

#Clase para el conjunto de datos
class data_set:
    def __init__(self, validation_set, test_set, fold):
        self.validation_set = validation_set
        self.test_set = test_set
        self.fold = fold

class final_dset:
    def __init__(self, test_set, train_set):
         self.test_set = test_set
         self.train_set = train_set

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
#Estocásticos sin escalado:
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

def SGD_GradoN(data_set, eta, iter, n):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train
        #Conversión a polinomio de grado N
        polynomial_features= PolynomialFeatures(degree=n)
        x_poly = polynomial_features.fit_transform(x)

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
    
    if n==2:
        exps_g2_mse.append(mse_prom)
        exps_g2_r2.append(r2_prom)
    if n==3:
        exps_g3_mse.append(mse_prom)
        exps_g3_r2.append(r2_prom)


#Estocásticos escalado estándar:
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
 
        x_poly_standard_scaler = preprocessing.StandardScaler().fit_transform(x)
 
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

def SGD_GradoN_StdScaler(data_set, eta, iter, n):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train

        polynomial_features= PolynomialFeatures(degree=n)
        x_poly = polynomial_features.fit_transform(x)
        x_poly_standard_scaler= preprocessing.StandardScaler().fit_transform(x_poly)

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
  
    if n==2:
        exps_g2_std_mse.append(mse_prom)
        exps_g2_std_r2.append(r2_prom)
    if n==3:
        exps_g3_std_mse.append(mse_prom)
        exps_g3_std_r2.append(r2_prom)


#Estocásticos escalados robusto
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
   
        x_poly_robust_scaler= preprocessing.RobustScaler().fit_transform(x)

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

def SGD_GradoN_RobScaler(data_set, eta, iter, n):
    print("\nLearning rate: constant; eta0: {}; iteraciones: {}".format(eta, iter))    

    mse_sum=0
    mse_prom=0
    r2_sum=0
    r2_prom=0
    i=1

    for val_set in data_set.validation_set:
        x= val_set.x_train
        y= val_set.y_train

        polynomial_features= PolynomialFeatures(degree=n)
        x_poly = polynomial_features.fit_transform(x)
        x_poly_robust_scaler= preprocessing.RobustScaler().fit_transform(x_poly)

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
    
    if n==2:
        exps_g2_rob_mse.append(mse_prom)
        exps_g2_rob_r2.append(r2_prom)
    if n==3:
        exps_g3_rob_mse.append(mse_prom)
        exps_g3_rob_r2.append(r2_prom)

#Entrenamiento y prueba final
def final_train(file_name, eta, iter, n):
    print("\nMejor resultado obtenido: Regresión polinomial estocástica de grado 3 con escalado estándar (del experimento 2).")
    pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a un DataFrame
    df = pd.read_csv(file_name, sep=',', engine='python')
    #Usa todas la columnas excepto medianHouseValue
    x = df.drop(['medianHouseValue'],axis=1).values   
    #Solamente usa la columna medianHouseValue
    y = df['medianHouseValue'].values
	
	#Separa el corpus cargado en el DataFrame en el 80% para entrenamiento y el 20% para pruebas
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True, random_state=0)

    #Asigna los test set y train set respectivamente
    my_test_set = test_set(x_test, y_test)
    my_train_set = train_set(x_train, y_train)

    #Crea el data set final para entrenar y probar
    my_data_set= final_dset(my_test_set, my_train_set)

    #Datos de entrenamiento
    x_tr= my_data_set.train_set.x_train
    y_tr= my_data_set.train_set.y_train

    #Datos de prueba
    x_te= my_data_set.test_set.x_test
    y_te= my_data_set.test_set.y_test

    polynomial_features= PolynomialFeatures(degree= n)
    #Entrenamiento del modelo
    print("\nEntrenando con el train set...")
    x_poly = polynomial_features.fit_transform(x_tr)
    x_poly_standard_scaler= preprocessing.StandardScaler().fit_transform(x_poly)
    
    regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter= iter)
    regr.fit(x_poly_standard_scaler, y_tr)
    #y_poly_pred = regr.predict(x_poly_standard_scaler)
    #mse = mean_squared_error(y_tr, y_poly_pred)
    #r2 = r2_score(y_tr, y_poly_pred)
    
    #print("\n*******************")
                
    #print("\nResultado del train set:\n\tmse: {}\n\tr2: {}".format(mse, r2))
    print("Entrenamiento terminado.")
    
    #Modelo para probar el test set
    print("\nProbando test set...")
    x_poly = polynomial_features.fit_transform(x_te)
    x_poly_standard_scaler= preprocessing.StandardScaler().fit_transform(x_poly)

    regr = SGDRegressor(learning_rate = 'constant', eta0 = eta, max_iter= iter)
    regr.fit(x_poly_standard_scaler, y_te)
    y_poly_pred = regr.predict(x_poly_standard_scaler)
    mse = mean_squared_error(y_te, y_poly_pred)
    r2 = r2_score(y_te, y_poly_pred)

    #print("\n*******************")

    #print('\ny_test: {}\n\ty_pred: {}'.format(y_te, y_poly_pred))
    print("\nResultados del test set:\n\tmse: {}\n\tr2: {}".format(mse, r2))

    #Lo siguiente decidí hacerlo para poder crear un csv 
    # para comparar los datos con los resultados predichos
    datos=[[],[]]
    
    y_te=y_te[:, np.newaxis]
    y_poly_pred= y_poly_pred[:, np.newaxis]
    
    datos[0]= y_te
    datos[1]= y_poly_pred
    datos=np.column_stack([datos[0], datos[1]])    

    #print("dato: {}".format(datos))

    np.savetxt("prediccion.csv", datos, delimiter=",", fmt="%s", header="medianHouseValue, prediction", comments="")
    
    """
    np.savetxt("datos.csv", my_data_set.test_set.y_test, delimiter=",", fmt="%s",
           header="medianHouseValue", comments="")
    np.savetxt("prediccion.csv", y_poly_pred, delimiter=",", fmt="%s",
           header="Predicted", comments="")	
    """
    

if __name__=='__main__':
    #Cantidad de pliegues
    k=10    		
    my_data_set = generate_train_test('cal_housing.csv',k)
    #Normalmente esto debería estar hasta abajo de todos esos comentarios siguientes
    # pero decidí subirlo para facilitar la lectura
    final_train('cal_housing.csv', 1e-5, 70000, 3)

    """
    print('\nDATOS TEST:')    
    print(my_data_set.test_set.x_test)
    print('\nDATOS A PREDECIR TEST:')
    print(my_data_set.test_set.y_test)
    print ('\n----------------------------------------------------------------------------------\n')
    """
    
    """
    #Hecho
    print("\n===== Regresión estocástico lineal sin escalar =====")
    print("\n---------------Lineal Experimento 1-----------------------------------")
    SGD_lineal(my_data_set, 0.000000001, 3000)

    print("\n---------------Lineal Experimento 2-----------------------------------")
    SGD_lineal(my_data_set, 0.000000005, 3000)

    print("\n---------------Lineal Experimento 3-----------------------------------")
    SGD_lineal(my_data_set, 0.0000000002, 3000)
    """
    """
    #Hecho (imposible positivo)
    print("\n===== Regresión polinomial estocástico grado 2 sin escalar =====")
    print("\n---------------Grado 2 Experimento 1-----------------------------------")
    SGD_GradoN(my_data_set, 9.9e-19, 20000, 2)

    print("\n---------------Grado 2 Experimento 2-----------------------------------")
    SGD_GradoN(my_data_set, 1e-19, 20000, 2)

    print("\n---------------Grado 2 Experimento 3-----------------------------------")
    SGD_GradoN(my_data_set, 5e-19, 20000, 2)
    """
    
    """
    #Hecho
    print("\n===== Regresión polinomial estocástico grado 3 sin escalar =====")
    print("\n---------------Grado 3 Experimento 1-----------------------------------")
    SGD_GradoN(my_data_set, 1e-19, 150000, 3)
    
    print("\n---------------Grado 3 Experimento 2-----------------------------------")
    SGD_GradoN(my_data_set, 1e-20, 150000, 3)

    print("\n---------------Grado 3 Experimento 3-----------------------------------")
    SGD_GradoN(my_data_set, 9e-19, 150000, 3)
    """
    
    """
    #Hecho
    print("\n===== Regresión estocástico lineal escalado estándar =====")
    print("\n---------------Lineal Estándar Experimento 1-----------------------------------")
    SGD_lineal_StdScaler(my_data_set, 0.000001, 10000)

    print("\n---------------Lineal Estándar Experimento 2-----------------------------------")
    SGD_lineal_StdScaler(my_data_set, 0.000009, 10000)
    
    print("\n---------------Lineal Estándar Experimento 3-----------------------------------")
    SGD_lineal_StdScaler(my_data_set, 0.0000002, 50000)
    
    
    #Hecho
    print("\n===== Regresión polinomial estocástico grado 2 escalado estándar =====")
    print("\n---------------Grado 2 Estándar Experimento 1-----------------------------------")
    SGD_GradoN_StdScaler(my_data_set, 0.0000008, 50000, 2)

    print("\n---------------Grado 2 Estándar Experimento 2-----------------------------------")
    SGD_GradoN_StdScaler(my_data_set, 0.0000009, 50000, 2)
    
    print("\n---------------Grado 2 Estándar Experimento 3-----------------------------------")
    SGD_GradoN_StdScaler(my_data_set, 1e-6, 50000, 2)
    
    
    #Hecho
    print("\n===== Regresión polinomial estocástico grado 3 escalado estándar =====")
    print("\n---------------Grado 3 Estándar Experimento 1-----------------------------------")
    SGD_GradoN_StdScaler(my_data_set, 1e-6, 70000, 3)

    print("\n---------------Grado 3 Estándar Experimento 2-----------------------------------")
    SGD_GradoN_StdScaler(my_data_set, 1e-5, 70000, 3)
    
    print("\n---------------Grado 3 Estándar Experimento 3-----------------------------------")
    SGD_GradoN_StdScaler(my_data_set, 5e-6, 70000, 3)
    """
    
    """
    #Hecho
    print("\n===== Regresión estocástico lineal escalado robusto =====")
    print("\n---------------Lineal Robusto Experimento 1-----------------------------------")
    SGD_lineal_RobScaler(my_data_set, 0.00001, 700000)

    print("\n---------------Lineal Robusto Experimento 2-----------------------------------")
    SGD_lineal_RobScaler(my_data_set, 0.00009, 700000)

    print("\n---------------Lineal Robusto Experimento 3-----------------------------------")
    SGD_lineal_RobScaler(my_data_set, 0.000002, 700000)
    
    #Hecho
    print("\n===== Regresión polinomial estocástico grado 2 escalado robusto =====")
    print("\n---------------Grado 2 Robusto Experimento 1-----------------------------------")
    SGD_GradoN_RobScaler(my_data_set, 0.000008, 700000, 2)

    print("\n---------------Grado 2 Robusto Experimento 2-----------------------------------")
    SGD_GradoN_RobScaler(my_data_set, 0.000009, 700000, 2)

    print("\n---------------Grado 2 Robusto Experimento 3-----------------------------------")
    SGD_GradoN_RobScaler(my_data_set, 1e-5, 700000, 2)
    """
    """
    #Hecho
    print("\n===== Regresión polinomial estocástico grado 3 escalado robusto =====")
    print("\n---------------Grado 3 Robusto Experimento 1-----------------------------------")
    SGD_GradoN_RobScaler(my_data_set, 0.000000009, 2000000, 3)

    print("\n---------------Grado 3 Robusto Experimento 2-----------------------------------")
    SGD_GradoN_RobScaler(my_data_set, 0.000000003, 3000000, 3)
    
    print("\n---------------Grado 3 Robusto Experimento 3-----------------------------------")
    SGD_GradoN_RobScaler(my_data_set, 0.000000005, 4000000, 3)
    """
    """
    print("\n\nResumen:")
    print("\nResultados SGD lineal:\n\tmse: {}\n\tr2: {}".format(exps_lineal_mse, exps_lineal_r2))
    print("\nResultados SGD Grado 2:\n\tmse: {}\n\tr2: {}".format(exps_g2_mse, exps_g2_r2))
    print("\nResultados SGD Grado 3:\n\tmse: {}\n\tr2: {}".format(exps_g3_mse, exps_g3_r2))

    print("\nResultados SGD lineal escalado estándar:\n\tmse: {}\n\tr2: {}".format(exps_lineal_std_mse, exps_lineal_std_r2))
    print("\nResultados SGD Grado 2 escalado estándar:\n\tmse: {}\n\tr2: {}".format(exps_g2_std_mse, exps_g2_std_r2))
    print("\nResultados SGD Grado 3 escalado estándar:\n\tmse: {}\n\tr2: {}".format(exps_g3_std_mse, exps_g3_std_r2))

    print("\nResultados SGD lineal escalado robusto:\n\tmse: {}\n\tr2: {}".format(exps_lineal_rob_mse, exps_lineal_rob_r2))
    print("\nResultados SGD Grado 2 escalado robusto:\n\tmse: {}\n\tr2: {}".format(exps_g2_rob_mse, exps_g2_rob_r2))
    print("\nResultados SGD Grado 3 escalado robusto:\n\tmse: {}\n\tr2: {}".format(exps_g3_rob_mse, exps_g3_rob_r2))
    """

    
