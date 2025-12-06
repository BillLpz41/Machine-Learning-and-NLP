# Programa para la deteccion de genero, profesion, ideologia binaria e ideologia multiclase apartir de tweets
from ast import Or
import re
import pandas as pd
import numpy as np
import spacy
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score as sklearn_f1_score
import pickle
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler

# Funcion para la normalizacion de los tweets
def normalizar_corpus(nombre_corpus, tipo):
    df= pd.read_csv(nombre_corpus, encoding='utf-8', sep=',', engine='python')
    tweets=df['tweet'].values
    categorias = df[['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass']].values

    cleaned=[]
    for tw in tweets:
        tw.strip()
        aux=re.sub(r'(@user\s?)+|(\[|\])+|[\U0000200D-\U0001FFFF]', '', tw)
        aux=re.sub(r'(\d{1,2}[-/\.]\d{1,2}([-/\.]\d{2,4}\b)?)|(\d{1,2}\s(d|de)\s(([eE]nero)|([fF]ebrero)|([mM]arzo)|([aA]bril)|([mM]ayo)|([jJ]unio)|([jJ]ulio)|([aA]gosto)|([sS]eptiembre)|([oO]ctubre)|([Nn]oviembre)|([Dd]iciembre))((\sde|\sdel)\d{2,4}\b)?)|(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b)|(\d{1,2}[-/\.][A-Za-z]{3}[-/\.]\d{2,4}\b)', '', aux)
        tw.strip()
        cleaned.append(aux)

    nlp = spacy.load("es_core_news_sm")
    stopwords= ['PRON', 'ADP', 'DET', 'CCONJ', 'PUNCT', 'NUM']

    # Inicializar arreglo de tweets normalizados
    tweets_norm=[]

    print("Normalizando tweets...")

    for tweets in cleaned:
        txt_norm=[] 
        doc= nlp(tweets)
        for token in doc:
            if token.tag_ not in stopwords:
                txt_norm.append(token.lemma_)
        tweets_norm.append(' '.join(txt_norm))

    # Combinar las columnas de y y x en un nuevo dataframe
    nuevo_df = np.column_stack((categorias, tweets_norm))

    # Convertir el nuevo dataframe a un objeto pandas y guardar como archivo .csv
    nuevo_df = pd.DataFrame(nuevo_df, columns=['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass', 'tweet'])

    # Condicion para elegir el tipo de agrupacion de tweets
    if tipo == 1:
        # Agrupar tweets por autor(Label)
        grouped = nuevo_df.groupby(['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass'])['tweet'].apply(list)
        agrupado = pd.DataFrame({'label': grouped.index.get_level_values(0),
                         'gender': grouped.index.get_level_values(1),
                         'profession': grouped.index.get_level_values(2),
                         'ideology_binary': grouped.index.get_level_values(3),
                         'ideology_multiclass': grouped.index.get_level_values(4),
                         'tweet': grouped.values})
    if tipo == 2:
        # Agrupar tweets por autor(Label) sin clasificacion de genero, profesion e ideologia
        grouped = nuevo_df.groupby(['label'])['tweet'].apply(list)
        agrupado = pd.DataFrame({'label': grouped.index.get_level_values(0),
                            'tweet': grouped.values})
    # Guardar dataframe en un archivo csv con nombre personalizado
    agrupado.to_csv('normalizado_'+str(nombre_corpus), index=False, encoding='utf-8')
    # Mensaje de finalizacion de la normalizacion
    print(str(nombre_corpus)+": Tweets normalizados con exito")

# Funcion para hacer prueba de parametros con grindsearch para regresion logistica
def Regresion_logistica_gridSearch(X_train, y_train, X_test, y_test):
    print("Entrenando modelo de regresion logistica...")
    # Crear modelo de regresion logistica
    #'C', 19.52548401046396), ('max_iter', 50)])
    modelo = LogisticRegression(class_weight="balanced", random_state=0, solver='saga')
    #Lista de parametros para probar con gridsearch
    parametros = {
    'C': (0.001, 150.0, 'log-uniform'),
    'max_iter': [100,300]
}
    # Usar gridsearch para probar parametros del modelo con f1_macro, 15 folds y acurracy
    grid = BayesSearchCV(modelo, parametros, cv=15, scoring='f1_macro', n_iter=30, n_jobs=5, verbose=3)
    # Entrenar modelo con gridsearch
    grid.fit(X_train, y_train)

    # Imprimir todos los parametos uitlizados
    print("--------------->Resultados de los parametros 80% train\n")
    print("Parametros probados: " + str(parametros))
    print("Mejores parametros: ", grid.best_params_)
    print("Mejor score para el 80 de entrenamiento: ", grid.best_score_)

    # Guardar los mejores parametros
    best = grid.best_estimator_
    return best
# Funcion parametros con gridsearch para maquina de soporte vectorial
def SVM_gridSearch(X_train, y_train, X_test, y_test):   
    print("Entrenando modelo de maquina de soporte vectorial...")
    # Crear modelo de maquina de soporte vectorial
    #SVC(kernel='linear', random_state=0, gamma=0.7, C=5.9, probability=True)
    modelo = SVC(kernel='linear', random_state=0, probability=True)
    #Lista de parametros para probar con gridsearch
    parametros = {'C': (0.01, 5.0, 'log-uniform'),
                  'gamma': (0.01, 5.0, 'log-uniform'),
    }
    # Usar gridsearch para probar parametros del modelo con f1_macro, 15 folds y acurracy
    grid = BayesSearchCV(modelo, parametros, cv=15, scoring='f1_macro', n_jobs=5, n_iter=5, verbose=3)
    # Entrenar modelo con gridsearch
    grid.fit(X_train, y_train)

    # Imprimir todos los parametos uitlizados
    print("--------------->Resultados de los parametros 80% train\n")
    print("Parametros probados: " + str(parametros))
    print("Mejores parametros: ", grid.best_params_)
    print("Mejor score para el 80 de entrenamiento: ", grid.best_score_)

    # Guardar los mejores parametros
    best = grid.best_estimator_
    return best
# Funcion parametros con gridsearch para perceptron multicapa
def MLP_gridSearch(X_train, y_train, X_test, y_test):
    print("Entrenando modelo de perceptron multicapa...")
    # Crear modelo de perceptron multicapa
    modelo = MLPClassifier(random_state=0, verbose=False, max_iter=300)
    #Lista de parametros para probar con gridsearch
    parametros = {'alpha': [0.19, 0.195, 0.2],
                  'hidden_layer_sizes': [(60), (70), (80)]
    }
    # Usar gridsearch para probar parametros del modelo con f1_macro, 15 folds y acurracy
    grid = GridSearchCV(modelo, parametros, cv=15, scoring='f1_macro', n_jobs=-1, verbose=3)
    # Entrenar modelo con gridsearch
    grid.fit(X_train, y_train)

    # Imprimir todos los parametos uitlizados
    print("--------------->Resultados de los parametros 80% train\n")
    print("Parametros probados: " + str(parametros))
    print("Mejores parametros: ", grid.best_params_)
    print("Mejor score para el 80 de entrenamiento: ", grid.best_score_)

    # Guardar los mejores parametros
    best = grid.best_estimator_
    return best

# Funcion para entrenar un modelo con los mejores aprametros y probarlo con el 20% de los datos
def mejor_parametro_gridSearch(best, X_train, y_train, X_test, y_test):     
    # Entrenar el 80% de los datos con los mejores parametros
    best.fit(X_train, y_train)

    # Hacer prediccion con el modelo entrenado
    # Prediccion del 20% de los datos
    y_pred = best.predict(X_test)    

    # Calcular la exactitud del modelo sobre el conjunto de prueba
    accuracy = accuracy_score(y_test, y_pred)
    # Calcular f1-score del modelo sobre el conjunto de prueba
    f1_score = sklearn_f1_score(y_test, y_pred, average='macro')
    # Calcular el classification report del modelo sobre el conjunto de prueba
    report = classification_report(y_test, y_pred)

    print("Exactitud del modelo 20% test:", accuracy)
    print("F1-Score del modelo 20% test:", f1_score)
    print("Classification report del modelo sobre el conjunto de prueba:")
    print(report)

    

# Funcion para comprobar si el modelo de reduccion de dimensionalidad ya existe
def comprobar_svd(X_train_simple,svd_modelo, tipo_vectorizacion):
    if os.path.isfile(svd_modelo):
        print("Cargando modelo de reduccion de dimensionalidad...")
        # Cargar modelo de reduccion de dimensionalidad
        with open(svd_modelo, 'rb') as f:
          svd = pickle.load(f)
        return svd
    # Si no existe lo crea
    else:
        # Representacion vectorial de los tweets
        X_train = tipo_vectorizacion.fit_transform(X_train_simple)
        # Crear modelo de reduccion de dimensionalidad
        svd = TruncatedSVD(n_components=950, random_state=0)
        # Entrenar modelo de reduccion de dimensionalidad
        New_svd = svd.fit_transform(X_train)
        # Guardar modelo de reduccion de dimensionalidad
        with open(svd_modelo, 'wb') as f:
            pickle.dump(New_svd, f)
        return New_svd
    
# Funcion para comprobar si los modelos de SVD para test y results ya existen
def comprobar_svd_test(X_train_Text, X_test_Text, X_results_Text, svd_modelo_train,svd_modelo_test, svd_modelo_result, tipo_vectorizacion):
    if os.path.isfile(svd_modelo_test) and os.path.isfile(svd_modelo_result):
        print("Cargando modelo de reduccion de dimensionalidad...")
        # Cargar modelo de reduccion de dimensionalidad
        with open(svd_modelo_train, 'rb') as f:
            svd1 = pickle.load(f)
        with open(svd_modelo_test, 'rb') as f:
            svd2 = pickle.load(f)
        with open(svd_modelo_result, 'rb') as f:
            svd3 = pickle.load(f)
        return svd1, svd2, svd3
    
    # Si no existe lo crea
    else:
        # Representacion vectorial de los tweets
        X_train = tipo_vectorizacion.fit_transform(X_train_Text)
        X_test = tipo_vectorizacion.transform(X_test_Text)
        X_results = tipo_vectorizacion.transform(X_results_Text)
        # Crear modelo de reduccion de dimensionalidad
        svd = TruncatedSVD(n_components=950, random_state=0)
        svd_train = svd.fit_transform(X_train)
        svd_test = svd.transform(X_test)
        svd_result = svd.transform(X_results)
        # Guardar modelo de reduccion de dimensionalidad
        with open(svd_modelo_train, 'wb') as f:
            pickle.dump(svd_train, f)
        with open(svd_modelo_test, 'wb') as f:
            pickle.dump(svd_test, f)
        with open(svd_modelo_result, 'wb') as f:
            pickle.dump(svd_result, f)
        return svd_train, svd_test, svd_result
    
# Funcion para dividir corpus en entrenamiento y prueba 80% y 20% respectivamente
def prueba_parametros(train_corpus, categoria, tipo_vectorizacion, svd_modelo):
    df = pd.read_csv(train_corpus, encoding='utf-8', sep=',', engine='python')
    # Extraer tweets y categorias
    tweets = df['tweet'].values
    categorias = df[categoria].values

    # Nombre que deberia tener el archivo SVD en caso de existir
    svd_modelo = str(svd_modelo)+"_train.pkl"

    # Comprobar si el modelo de reduccion de dimensionalidad ya existe
    X_train_svd = comprobar_svd(tweets, svd_modelo, tipo_vectorizacion)
    
    # Divir el corpus de entrenamiento en 80% y 20%
    X_train, X_test, y_train, y_test = train_test_split(X_train_svd, categorias, test_size=0.2, random_state=0)
    # Funciones para probar parametros de los modelos
    
    # Regresion logistica
    
    modelo1 = Regresion_logistica_gridSearch(X_train, y_train, X_test, y_test)
    print("\nAplicar los mejores parametros...\n")
    mejor_parametro_gridSearch(modelo1, X_train, y_train, X_test, y_test)
    """
    # Maquina de soporte vectorial
    modelo2 = SVM_gridSearch(X_train, y_train, X_test, y_test)
    print("\nAplicar los mejores parametros...\n")
    mejor_parametro_gridSearch(modelo2, X_train, y_train, X_test, y_test)
    
    # Perceptron multicapa
    modelo3 = MLP_gridSearch(X_train, y_train, X_test, y_test)
    print("\nAplicar los mejores parametros...\n")
    mejor_parametro_gridSearch(modelo3, X_train, y_train, X_test, y_test)
    """
# Funcion para realizar el entrenamiento usando votacion de modelos
def entrenamiento_modelos(train_corpus, voting_clf, categoria, tipo_normalizacion, svd_modelo):
    # Cargar los tweets y categorias a un dataframe
    df = pd.read_csv(train_corpus, encoding='utf-8', sep=',', engine='python')
    # Extraer tweets y categorias
    tweets = df['tweet'].values
    categorias = df[categoria].values

    # Nombre que deberia tener el archivo SVD en caso de existir
    svd_modelo = str(svd_modelo)+"_train.pkl"

    # Comprobar si el modelo de reduccion de dimensionalidad ya existe
    X_train_svd = comprobar_svd(tweets, svd_modelo, tipo_normalizacion)

    # Divir el corpus de entrenamiento en 80% y 20%
    X_train, X_test, y_train, y_test = train_test_split(X_train_svd, categorias, test_size=0.2, random_state=0)

    # Volting de modelos
    print("Entrenando modelo de votacion de modelos...")
    print("--------------Para 80% de entrenamiento--------------\n")
    # Entrenar modelo de votacion de modelos usando validacion cruzada de 15 folds
    scores = cross_val_score(voting_clf, X_train, y_train, cv=15, scoring='f1_macro', n_jobs=5, verbose=3)

    # Imprimir resultados de la validacion cruzada
    print("Resultados de la validacion cruzada 80 entrenamiento:")
    print(scores)
    print("Promedio de la validacion cruzada 80 entrenamiento:", scores.mean())

    # Pruebas de los modelos con el 20% de los datos
    print("--------------Para 20% de prueba--------------\n")
    # Entrenar modelo de votacion de modelos
    voting_clf.fit(X_train, y_train)
    # Hacer prediccion con el modelo entrenado
    y_pred = voting_clf.predict(X_test)
    # Calcular la exactitud del modelo sobre el conjunto de prueba
    accuracy = accuracy_score(y_test, y_pred)
    # Calcular f1-score del modelo sobre el conjunto de prueba
    f1_score = sklearn_f1_score(y_test, y_pred, average='macro')
    # Calcular el classification report del modelo sobre el conjunto de prueba
    report = classification_report(y_test, y_pred)

    print("Exactitud del modelo 20% test:", accuracy)
    print("F1-Score del modelo 20% test:", f1_score)
    print("Classification report del modelo sobre el conjunto de prueba:")
    print(report)

    # Crear la visualizacion de la matriz de confusion
    sns.set(font_scale=1.4)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g')
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.title('Matriz de confusion')
    # Si la categoria corresponde a gender o ideology_binary
    if (categoria == 'gender' or categoria == 'ideology_binary'):
        plt.text(0, 2.5, f"Accuracy: {accuracy:.3f}", fontsize=15)
        plt.text(0, 2.7, f"F1-Score: {f1_score:.3f}", fontsize=15)
        plt.text(0, 3.8, f"Classification Report:\n{report}", fontsize=15)
        # Guardar la imagen de la matriz de confusion
        plt.savefig('Resultados_entrenamiento-20_'+str(categoria)+'.png', bbox_inches='tight')

    # Si la categoria corresponde a profession aplica los siguientes cambios
    elif categoria == 'profession':
        plt.text(0, 3.6, f"Accuracy: {accuracy:.3f}", fontsize=15)
        plt.text(0, 3.8, f"F1-Score: {f1_score:.3f}", fontsize=15)
        plt.text(0, 5.4, f"Classification Report:\n{report}", fontsize=15)
        # Guardar la imagen de la matriz de confusion
        plt.savefig('Resultados_entrenamiento-20_'+str(categoria)+'.png', bbox_inches='tight')

    # Si la categoria corresponde a ideology_multiclass aplica los siguientes cambios
    elif categoria == 'ideology_multiclass':
        plt.text(0, 4.8, f"Accuracy: {accuracy:.3f}", fontsize=15)
        plt.text(0, 5.1, f"F1-Score: {f1_score:.3f}", fontsize=15)
        plt.text(0, 7.5, f"Classification Report:\n{report}", fontsize=15)

        # Guardar la imagen de la matriz de confusion
        plt.savefig('Resultados_entrenamiento-20_'+str(categoria)+'.png', bbox_inches='tight')


# Funcion para hacer predicciones sobre el conjunto de pruebas y de resultados
def generar_results(train_corpus, test_corpus, results_corpus, voting_clf, categoria, tipo_vectorizacion, svd_modelo):
    # Cargar los tweets y categorias a un dataframe del conjunto de entrenamiento
    df = pd.read_csv(train_corpus, encoding='utf-8', sep=',', engine='python')
    # Extraer tweets y categorias
    tweets = df['tweet'].values
    categorias = df[categoria].values

    # Cargar los tweets y categorias a un dataframe del conjunto de prueba
    df2 = pd.read_csv(test_corpus, encoding='utf-8', sep=',', engine='python')
    # Extraer tweets y categorias
    tweets2 = df2['tweet'].values
    categorias2 = df2[categoria].values

    # Cargar los tweets y categorias a un dataframe del conjunto de resultados
    df3 = pd.read_csv(results_corpus, encoding='utf-8', sep=',', engine='python')
    # Extraer tweets y categorias
    tweets3 = df3['tweet'].values

    # Nombre que deberia tener el archivo SVD en caso de existir
    svd_modelo_train = str(svd_modelo)+"_train.pkl"
    svd_modelo_test = str(svd_modelo)+"_test.pkl"
    svd_modelo_result = str(svd_modelo)+"_result.pkl"

    # Comprobar si los modelo de reduccion de dimensionalidad ya existen
    X_train_svd, X_test_svd, X_results_svd = comprobar_svd_test(tweets, tweets2, tweets3, svd_modelo_train, svd_modelo_test, svd_modelo_result, tipo_vectorizacion)

    # Entrenar el modelo de votacion de modelos con el conjunto completo de datos
    voting_clf.fit(X_train_svd, categorias)

    # Genero una prediccion con el modelo entrenado sobre el conjunto de prueba
    y_pred = voting_clf.predict(X_test_svd)


    # Obtener los nombres de las categorias
    target_names = voting_clf.classes_
    # Calculo la exactitud del modelo sobre el conjunto de prueba
    accuracy = accuracy_score(categorias2, y_pred)
    # Calculo f1-score del modelo sobre el conjunto de prueba
    f1_score = sklearn_f1_score(categorias2, y_pred, average='macro')
    # Calculo el classification report del modelo sobre el conjunto de prueba
    report = classification_report(categorias2, y_pred)
    # Calculo la matriz de confusion del modelo sobre el conjunto de prueba
    cm = confusion_matrix(categorias2, y_pred, labels=target_names)
    #Muestra la matriz de confusión
    confusion_mat = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    print("--------------Para el conjunto de pruebas--------------\n")
    print("Exactitud del modelo 20% test:", accuracy)
    print("F1-Score del modelo 20% test:", f1_score)
    print("Classification report del modelo sobre el conjunto de prueba:")
    print(report)

    # Crear la visualizacion de la matriz de confusion
    sns.set(font_scale=1.4)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(categorias2, y_pred), annot=True, fmt='g')
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.title('Matriz de confusion')
    # Si la categoria corresponde a gender o ideology_binary
    if (categoria == 'gender' or categoria == 'ideology_binary'):
        plt.text(0, 2.5, f"Accuracy: {accuracy:.3f}", fontsize=15)
        plt.text(0, 2.7, f"F1-Score: {f1_score:.3f}", fontsize=15)
        plt.text(0, 3.8, f"Classification Report:\n{report}", fontsize=15)
        # Guardar la imagen de la matriz de confusion
        plt.savefig('Resultados_pruebas_'+str(categoria)+'.png', bbox_inches='tight')

    # Si la categoria corresponde a profession aplica los siguientes cambios
    elif categoria == 'profession':
        plt.text(0, 3.6, f"Accuracy: {accuracy:.3f}", fontsize=15)
        plt.text(0, 3.8, f"F1-Score: {f1_score:.3f}", fontsize=15)
        plt.text(0, 5.4, f"Classification Report:\n{report}", fontsize=15)
        # Guardar la imagen de la matriz de confusion
        plt.savefig('Resultados_pruebas_'+str(categoria)+'.png', bbox_inches='tight')

    # Si la categoria corresponde a ideology_multiclass aplica los siguientes cambios
    elif categoria == 'ideology_multiclass':
        plt.text(0, 4.8, f"Accuracy: {accuracy:.3f}", fontsize=15)
        plt.text(0, 5.1, f"F1-Score: {f1_score:.3f}", fontsize=15)
        plt.text(0, 7.5, f"Classification Report:\n{report}", fontsize=15)

        # Guardar la imagen de la matriz de confusion
        plt.savefig('Resultados_pruebas_'+str(categoria)+'.png', bbox_inches='tight')

    # Genero una prediccion con el modelo entrenado sobre el conjunto de resultados
    y_pred_results = voting_clf.predict(X_results_svd)

    # Compruebo si ya existe un archivo donde guardar mis predicciones
    if os.path.isfile('results.csv'):
        # Abro el archivo existente
        df4 = pd.read_csv('results.csv', encoding='utf-8', sep=',', engine='python')
        # Guardo en la columna de categoria las predicciones
        df4[categoria] = y_pred_results
        # Guardo el archivo con las predicciones
        df4.to_csv('results.csv', index=False, encoding='utf-8')
    # Si no existe el archivo lo creo
    else:
        # Creo un dataframe con las columnas de label, gender, profession, ideology_binary, ideology_multiclass
        # Guardo en la columna de categoria las predicciones
        nuevo_df = pd.DataFrame(columns=['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass'])
        nuevo_df[categoria] = y_pred_results
        # Copio la columna de label del corpus de resultados al nuevo dataframe
        nuevo_df['label'] = df3['label']
        # Guardo el archivo con las predicciones
        nuevo_df.to_csv('results.csv', index=False, encoding='utf-8')
    
    # Regreso el valor de f1-score
    return f1_score

# Funcion para generar modelos de prediccion
def politics_model(train_corpus, test_corpus, results_corpus):
    """
    # Predicon de la clase genero
    print("---------------->Genero<-----------------\n")
    categoria = 'gender'
    svd_modelo = 'svd_'+str(categoria)
    #mis_pesos = [0.89, 0.90, 0.91]
    mis_pesos = [0.91,0.90]
    tipo_vectorizacion = TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
    print("Usando maquina de soporte vectorial y perceptron multicapa...\n")
    modelo3 = SVC(kernel='linear', random_state=0, gamma=0.10027634876406606, C=2.118800541277415, probability=True)
    modelo2 = MLPClassifier(random_state=0, verbose=False, hidden_layer_sizes=(60), max_iter=300, alpha=0.195)
    modelo1 = LogisticRegression(class_weight="balanced", random_state=0, solver='saga', penalty='l2', max_iter=250, C=15)
    #'C', 19.52548401046396), ('max_iter', 50)])
    volting = VotingClassifier(estimators=[('LG', modelo1), ('MLP', modelo2)], voting='soft', weights=mis_pesos, verbose=4)
    prueba_parametros(train_corpus, categoria, tipo_vectorizacion, svd_modelo)
    #entrenamiento_modelos(train_corpus, modelo2, categoria, tipo_vectorizacion, svd_modelo)
    #generar_results(train_corpus, test_corpus, results_corpus, volting, categoria, tipo_vectorizacion, svd_modelo)
    
    # Predicon de la clase profesion
    print("---------------->Profesion<-----------------\n")
    categoria = 'profession'
    svd_modelo = 'svd_'+str(categoria)
    #mis_pesos = [0.89, 0.90, 0.91]
    mis_pesos = [0.96,0.95,0.94]
    # Tipo de vectorizacion por frecuencia
    tipo_vectorizacion = CountVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
    print("Usando maquina de soporte vectorial y perceptron multicapa...\n")
    modelo3 = LogisticRegression(class_weight="balanced", random_state=0, solver='saga', penalty='l2', max_iter=500, C=0.1)
    modelo2 = SVC(kernel='linear', random_state=0, C=0.1, gamma=0.1, probability=True)
    modelo1 = MLPClassifier(random_state=0, verbose=False, max_iter=250, hidden_layer_sizes=(50), alpha=1.4)
    volting = VotingClassifier(estimators=[('MLP', modelo1), ('SVM', modelo2), ('RL', modelo3)], voting='soft', weights=mis_pesos, verbose=4)
    prueba_parametros(train_corpus, categoria, tipo_vectorizacion, svd_modelo)
    #entrenamiento_modelos(train_corpus, volting, categoria, tipo_vectorizacion, svd_modelo)
    #generar_results(train_corpus, test_corpus, results_corpus, volting, categoria, tipo_vectorizacion, svd_modelo)
    
    # Predicon de la clase ideologia binaria
    print("---------------->Ideologia binaria<-----------------\n")
    categoria = 'ideology_binary'
    svd_modelo = 'svd_'+str(categoria)
    #mis_pesos = [0.89, 0.90, 0.91]
    mis_pesos = [0.95,0.94]
    # Tipo de vectorizacion por frecuencia
    #tipo_vectorizacion = TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
    print("Usando maquina de soporte vectorial y perceptron multicapa...\n")
    modelo2=LogisticRegression(class_weight="balanced",random_state=0, C=0.1, solver='saga', penalty='l2', max_iter=450)
    modelo1=MLPClassifier(random_state=0, alpha= 1.5, hidden_layer_sizes= (20) , max_iter=300)
    modelo3=SVC(kernel='linear', random_state=0, C=0.1, gamma=0.1, probability=True)
    volting = VotingClassifier(estimators=[('MLP', modelo1), ('LR', modelo2)], voting='soft', weights=mis_pesos, verbose=4)
    #prueba_parametros(train_corpus, categoria, tipo_vectorizacion, svd_modelo)
    entrenamiento_modelos(train_corpus, volting, categoria, tipo_vectorizacion, svd_modelo)
    #generar_results(train_corpus, test_corpus, results_corpus, volting, categoria, tipo_vectorizacion, svd_modelo)
    """
    # Predicon de la clase ideologia multiclase
    print("---------------->Ideologia multiclase<-----------------\n")
    categoria = 'ideology_multiclass'
    svd_modelo = 'svd_'+str(categoria)
    #mis_pesos = [0.89, 0.90, 0.91]
    mis_pesos = [0.89,0.87,0.86]
    # Tipo de vectorizacion por frecuencia
    tipo_vectorizacion = TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
    print("Usando maquina de soporte vectorial y perceptron multicapa...\n")
    modelo2=LogisticRegression(class_weight="balanced", multi_class="multinomial", random_state=0, C=0.15, solver='saga', penalty='l2', max_iter=300)
    modelo1=MLPClassifier(random_state=0, alpha= 0.3, hidden_layer_sizes= (60) , max_iter=300)
    modelo3=SVC(kernel='linear', random_state=0, C=0.11, gamma=0.05, probability=True)
    volting = VotingClassifier(estimators=[('MLP', modelo1), ('LR', modelo2), ('SVM', modelo3)], voting='soft', weights=mis_pesos, verbose=4)
    prueba_parametros(train_corpus, categoria, tipo_vectorizacion, svd_modelo)
    #entrenamiento_modelos(train_corpus, volting, categoria, tipo_vectorizacion, svd_modelo)
    #generar_results(train_corpus, test_corpus, results_corpus, volting, categoria, tipo_vectorizacion, svd_modelo)
    

# Funcion main
if __name__=='__main__':
    # Mensaje de bienvenida
    print("Bienvenido al programa de normalizacion de tweets")
    # Nombre y ubucacion de los archivos a normalizar
    entrenamiento_corpus = "Politic_Train.csv"
    prueba_corpus = "development_test.csv"
    resultados_corpus = "Politic_Test.csv"

    # Nombre de los corpus normalizados con nombre personalizado
    entrenamiento_normalizado = "normalizado_"+str(entrenamiento_corpus)
    prueba_normalizado = "normalizado_"+str(prueba_corpus)
    resultados_normalizado = "normalizado_"+str(resultados_corpus)

    # Iniciando la normalizacion de los corpus
    # Comprobar si los archivos normalizados ya existen
    if ((os.path.exists(prueba_normalizado)==False)):
        print("-------------Normalizando "+str(prueba_corpus)+"------------\n")
        # Funcion de normalizacion caso 1 (Si hay etiquetas)
        normalizar_corpus(prueba_corpus, 1)

    print("Cargando corpus de resultados "+str(resultados_normalizado)+"...\n")

    if ((os.path.exists(resultados_normalizado)==False)):
        print("-------------Normalizando "+str(resultados_corpus)+"------------\n")
        # Funcion de normalizacion caso 2 (No hay etiquetas)
        normalizar_corpus(resultados_corpus, 2)

    print("Crgando corpus de test "+str(prueba_normalizado)+"...\n")

    if ((os.path.exists(entrenamiento_normalizado)==False)):
        print("-------------Normalizando "+str(entrenamiento_corpus)+"------------\n")
        # Funcion de normalizacion caso 1 (Si hay etiquetas)
        normalizar_corpus(entrenamiento_corpus, 1)

    print("Cargando corpus de entrenamiento "+str(entrenamiento_normalizado)+"...\n")
    # Funcion para generar modelos de prediccion
    politics_model(entrenamiento_normalizado, prueba_normalizado, resultados_normalizado)
   