print("====== SIMILITUD DE DOCUMENTOS ===========")
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import math

def vector_frec(corpus, test):
	# Creación de un objeto CountVectorizer para el conteo de frecuencia de palabras
	v_frec= CountVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
	
	# Ajuste del vectorizador al corpus de entrenamiento y transformación de los datos de entrenamiento
	x= v_frec.fit_transform(corpus)

	# Ajuste del vectorizador al corpus de entrenamiento y transformación de los datos de prueba
	y= v_frec.fit(corpus)
	y= v_frec.transform(test)
	
	# Devolver los datos de entrenamiento y prueba transformados
	return x,y

def vector_bin(corpus, test):
	# Creación de un objeto CountVectorizer para la creación de vectores binarios
	v_bin= CountVectorizer(binary=True, token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
	
	# Ajuste del vectorizador al corpus de entrenamiento y transformación de los datos de entrenamiento
	x= v_bin.fit_transform(corpus)

	# Ajuste del vectorizador al corpus de entrenamiento y transformación de los datos de prueba
	y= v_bin.fit(corpus)
	y= v_bin.transform(test)
	
	# Devolver los datos de entrenamiento y prueba transformados
	return x,y

def vector_tfidf(corpus, test):
	# Creación de un objeto TfidfVectorizer para la creación de vectores ponderados
	v_tfidf= TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
	
	# Ajuste del vectorizador al corpus de entrenamiento y transformación de los datos de entrenamiento
	x= v_tfidf.fit_transform(corpus)

	# Ajuste del vectorizador al corpus de entrenamiento y transformación de los datos de prueba
	y= v_tfidf.fit(corpus)
	y= v_tfidf.transform(test)
	
	# Devolver los datos de entrenamiento y prueba transformados
	return x,y

def normalizar(test, stopw):
	# Cargar el modelo de procesamiento de lenguaje natural 'es_core_news_sm' de Spacy
	nlp= spacy.load('es_core_news_sm')
	
	# Procesar el texto de prueba con el modelo de Spacy
	doc=nlp(test)
	
	# Lista vacía para guardar el texto normalizado
	c_norm=[]
	
	# Recorrer cada token del texto procesado
	for x in doc:
		if x.tag_ not in stopw: # Si el token no es una palabra vacía o de detención, se lematiza y se agrega a la lista de texto normalizado
			c_norm.append(x.lemma_)
	
	# Unir los tokens normalizados con un espacio en blanco
	c_norm=' '.join(c_norm)
	
	# Devolver el texto normalizado
	return c_norm


def subirtxt(nombre):
    # Abre el archivo de texto con el nombre especificado
    txt= open(nombre,"r")

    # Lee el contenido del archivo de texto
    y= txt.read()

    # Llama a la función "normalizar" para normalizar el texto leído
    aux= normalizar(y, eliminar)

    # Crea una lista y añade el texto normalizado a la lista
    y_norm=[]
    y_norm.append(aux)

    # Devuelve la lista que contiene el texto normalizado
    return y_norm

def menu():
    # Imprime el menú en la consola
    print("\n¿Cuál representación del vectorizador deseas?")
    print("1. Vectorizar en frecuencia")
    print("2. Vectorizar en binario")
    print("3. Vectorizar con tf-idf")
    print("4. Finalizar programa")

    # Solicita al usuario que ingrese una opción del menú
    # "end=''" evita que se imprima un salto de línea después del mensaje
    print("\nIngresa el número de la opción que quieres ejecutar: ", end="")
    c= input()

    # Devuelve la opción elegida por el usuario como un entero
    return int(c)


if __name__=='__main__':
	# Lee el archivo CSV con pandas y almacena en una variable la columna 'noticia'
	df= pd.read_csv('texto_normalizado.csv', encoding='latin1', sep=',', engine='python', index_col=False)
	corpus_noticias = df['noticia']
	
	# Lee el titular de las noticias del archivo CSV y almacena en una variable
	titular_noticias =pd.read_csv('texto_normalizado.csv', usecols=['titular'], header=0)
	eliminar= ['PRON', 'ADP', 'DET', 'CCONJ']

	# Solicita al usuario el nombre del archivo que desea comparar y lo normaliza
	nombre= input("Escribe el nombre o la ruta del archivo a comparar (incluya la extesión del archivo): ")
	y_norm= subirtxt(nombre)

	# Crea una lista vacía para almacenar las similitudes coseno
	similitudes = []

	while True:
		c = menu()

		if c == 1:
			print("Vector de frecuencia\n")
			frec_corpus, frec_test = vector_frec(corpus_noticias, y_norm)
			print("Tabla de frecuencias X:\n")
			x_vectorizado = frec_corpus.toarray()
			print(x_vectorizado)
			print("\nTabla de frecuencias Y:\n")
			print(frec_test.toarray())

		elif c == 2:
			print("Vector binario\n")
			frec_corpus, frec_test = vector_bin(corpus_noticias, y_norm)
			print("Tabla de binarizado X:\n")
			print(frec_corpus.toarray())
			print("\nTabla de binarizado Y:\n")
			print(frec_test.toarray())

		elif c == 3:
			print("Vector tf-idf\n")
			frec_corpus, frec_test = vector_tfidf(corpus_noticias, y_norm)
			print("Tabla tf-idf X:\n")
			print(frec_corpus.toarray())
			print("\nTabla tf-idf X:\n")
			print(frec_test.toarray())

		elif c == 4:
			print("Finalizando...\n")
			print("=== PROGRAMA TERMINADO ===")
			sys.exit()
		else:
			print("Erraste, intenta de nuevo.\n")
		
		for idx, titular in titular_noticias.iterrows():
			# Obtiene el vector de frecuencias de la noticia actual
			frec_noticia_corpus = frec_corpus[idx, :]
	
			# Calcula el numerador y denominador de la similitud coseno
			numerador = np.dot(frec_noticia_corpus.toarray()[0], frec_test.toarray()[0])
			denominador = math.sqrt(sum(frec_noticia_corpus.toarray()[0]**2)) * math.sqrt(sum(frec_test.toarray()[0]**2))
			# Calcula la similitud coseno entre los dos documentos
			similitud = numerador / denominador

			# Almacena la similitud coseno correspondiente en la lista de similitudes
			similitudes.append((titular[0], similitud))
	
		# Ordena la lista de similitudes en orden descendente por similitud coseno y toma los primeros 10 elementos
		similitudes = sorted(similitudes, key=lambda x: x[1], reverse=True)[:10]

		# Imprime la lista de los 10 documentos más similares
		print("Los 10 documentos más similares al de prueba son:")
		for similitud in similitudes:
			print(f"- Documento '{similitud[0]}' con similitud coseno {similitud[1]}")

