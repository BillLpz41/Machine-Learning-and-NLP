print("====== SIMILITUD DE DOCUMENTOS ===========")
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import math
#conda install -c conda-forge spacy-model-es_core_news_sm
#python -m spacy download es_core_news_sm

#~ Funciones para los vectorizadores de frecuencia, binarización y tf-idf
def vector_frec(corpus, test):
	v_frec= CountVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
	x= v_frec.fit_transform(corpus)

	#y= v_frec.fit(corpus) #No es necesario, v_frec ya contiene el fit
	y= v_frec.transform(test)
	#print(y)
	return x,y

def vector_bin(corpus, test):
	v_bin= CountVectorizer(binary=True, token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
	x= v_bin.fit_transform(corpus)

	#y= v_bin.fit(corpus)
	y= v_bin.transform(test)
	return x,y

def vector_tfidf(corpus, test):
	v_tfidf= TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
	x= v_tfidf.fit_transform(corpus)

	#y= v_tfidf.fit(corpus)
	y= v_tfidf.transform(test)
	return x,y

#~ Función para normalizar el texto de prueba
def normalizar(test):
	c_norm=[]
	stopwords= ['PRON', 'ADP', 'DET', 'CCONJ']

	nlp= spacy.load('es_core_news_sm')
	doc=nlp(test)
	
	for x in doc:
		#print(x)
		if x.tag_ not in stopwords:
			c_norm.append(x.lemma_)
	#print(c_norm)
	#~ Eliminando las comas que separan los caracteres
	c_norm=' '.join(c_norm)
	print("Corpus de prueba normalizado exitosamente.")
	
	return c_norm

#~ Función para subir el archivo de texto
def subirtxt(nombre):
	txt= open(nombre,"r", encoding="utf8")
	#txt= np.loadtxt(nombre, dtype=str)
	#print(txt)
	y= txt.read()
	#y= pd.DataFrame(txt)
	#print(y)
	aux= normalizar(y)
	#print(aux)
	y_norm=[]
	y_norm.append(aux)
	#print(y_norm)
	return y_norm

def sim_coseno(x, y, titulares):
	simil=[]
	fila=1
	#print(y[0])
	for noticia, titulos in zip(x, titulares):
		#if(c==2):
		#	break
		numerador=0
		denominador=0
		sum_sqrt_x=0
		sum_sqrt_y=0

		#suma+= (j.item())*(k.item())
		numerador= np.dot(noticia, y[0])
		#print(numerador)
		sum_sqrt_x= sum(noticia**2)
		#print(sum_sqrt_x)
		sum_sqrt_y= sum(y[0]**2)	
		#print(sum_sqrt_y)
		denominador= (math.sqrt(sum_sqrt_x))*(math.sqrt(sum_sqrt_y))
		#print(denominador)
		aux= numerador/denominador
		#print(aux)
		simil.append((titulos, aux, fila))
		fila+=1
		
	# Ordena la lista de similitudes en orden descendente por similitud coseno y toma los primeros 10 elementos
	simil = sorted(simil, key=lambda x: x[1], reverse=True)[:10]

	# Imprime la lista de los 10 documentos más similares
	print("\nLos 10 documentos más similares al de prueba son:\n")
	for s, n in zip(simil, range(len(simil))):
		print(f"No. {n+1} || Fila: {s[2]} - Noticia: '{s[0]}' \n\tSimilitud coseno: {s[1]}\n")
	
#~ Función para escoger las opciones de vectorización y otras cosas
def menu():
	print("\n¿Cuál representación del vectorizador deseas?")
	print("1. Vectorizar en frecuencia")
	print("2. Vectorizar en binario")
	print("3. Vectorizar con tf-idf")
	print("4. Escoger otro archivo de texto")
	print("5. Finalizar programa")
	#~ end="" es para evitar el salto de línea
	print("\nIngresa el número de la opción que quieres ejecutar: ", end="")	
	c= input()

	return int(c)

if __name__=='__main__':
	print("Cargando corpus...")
	#~ El dataset estará codificado como latin1 para que el texto sea leíble
	df= pd.read_csv('noticia_normalizado.csv', encoding='latin1', sep=',', engine='python', header=None)
	x=df.values
	#print(x[0])

	titulos= pd.read_csv('texto_normalizado.csv', usecols=['titular'], header=0)
	t= titulos['titular'].values

	nombre= input("Escribe el nombre o la ruta del archivo que quieres comparar su similitud (incluya su extesión en txt): ")
	y_norm= subirtxt(nombre)

	while True:
		c= menu()

		if c == 1:
			print("Vector de frecuencia\n")
			#~ El método ravel() de numpy aplana el arreglo para que sea unidimensional
			x_norm, y_vector= vector_frec(x.ravel(), y_norm)

			print("Tabla de frecuencias X:\n")
			x_vectorizado= x_norm.toarray()
			print(x_vectorizado)
			print("\nTabla de frecuencias Y:\n")
			print(y_vector.toarray())

			sim_coseno(x_vectorizado, y_vector.toarray(), t)
			#print(similitud)

		elif c == 2:
			print("Vector binario\n")
			x_norm, y_vector= vector_bin(x.ravel(), y_norm)

			print("Tabla de binarizado X:\n")
			print(x_norm.toarray())
			print("\nTabla de binarizado Y:\n")
			print(y_vector.toarray())
			x_vectorizado= x_norm.toarray()
			sim_coseno(x_vectorizado, y_vector.toarray(), t)

		elif c == 3:
			print("Vector tf-idf\n")
			x_norm, y_vector= vector_tfidf(x.ravel(), y_norm)

			print("Tabla tf-idf X:\n")
			print(x_norm.toarray())
			print("\nTabla tf-idf X:\n")
			print(y_vector.toarray())
			x_vectorizado= x_norm.toarray()
			sim_coseno(x_vectorizado, y_vector.toarray(), t)

		elif c == 4:
			y_norm=[]
			nombre= input("\nEscribe el nombre o la ruta del archivo que quieres comparar su similitud (incluye su extesión en txt): ")
			y_norm= subirtxt(nombre)

		elif c == 5:
			print("Finalizando...\n")
			print("=== PROGRAMA TERMINADO ===")
			sys.exit()
			#break
		else:
			print("Erraste, intenta de nuevo.\n")
		
		x_vectorizado=0
		x_norm=0
		y_vector=0


#--------------------------
# CEMENTERIO DE CÓDIGO	
#--------------------------
	"""
	#~ Artículos, preposiciones, conjunciones y pronombes que se buscará eliminar
	

	print("Preprocesando texto...")
	tokens_titulos=[]
	tokens_content=[]

	tag_titulos=[]
	tag_contenido=[]

	lem_titulos=[]
	lem_cont=[]

	#t_label= np.array(t_label)
	#lem_t= np.array(tag_titulos)
	#lem_c= np.array(tag_contenido, dtype=object)

	#~ Almacena las listas en un arreglo 2D en forma de columnas
	t_lem_tokens= np.column_stack([txt_titulos, tag_titulos, lem_titulos])
	c_lem_tokens= np.column_stack([txt_cont, tag_contenido, lem_cont])
	print(t_lem_tokens)

	print("Generando dataframe con los tokens lematizados...")
	#~ Guarda el arreglo anterior en un dataframe de pandas con las columnas nombradas
	#datos= pd.DataFrame(lem_tokens, columns=['ID', 'Token_titulo y etiqueta', 'Token_Contenido y etiqueta'])
	t_lem= pd.DataFrame(t_lem_tokens, columns=['Token_titulo', 'Token_lematizado', 'Etiqueta'])
	t_lem= pd.DataFrame(t_lem_tokens, columns=['Token_noticia', 'Token_lematizado', 'Etiqueta'])

	#~ Guarda el Dataframe en un CSV, con el parámetro del índice desactivado (número de fila)
	# y la codificación en latin1 para que el texto tengo un formato leíble
	t_lem.to_csv('titulos_lematizados.csv', sep=",", index=False, encoding="latin1")
	t_lem.to_csv('noticias_lematizados.csv', sep=",", index=False, encoding="latin1")

	print("Archivo CSV con los tokens lematizados creado exitosamente.\n")

	print("====== FIN DE LA NORMALIZACIÓN DEL TEXTO ======")
	
	"""
#print(y[0])
#Un programa del equipo conformado por Billy, Arturo, Josué y Gustavo
#print(similitud)
"""
def test_frec(test, corpus):
	v_frec= CountVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
	y= v_frec.fit(corpus)
	print(corpus)
	y= v_frec.transform(test)
	print(y)
	return y

def test_bin(test, corpus):
	v_bin= CountVectorizer(binary= True, token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
	x= v_bin.fit(corpus)
	y= v_bin.transform(test)
	return y

def test_tfidf(test, corpus):
	v_tfidf= TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
	x= v_tfidf.fit(corpus)
	y= v_tfidf.transform(test)
	return y
"""
"""
def tokenizar(nuevo_corpus):
	for noticia in nuevo_corpus:
		tokens_content.append(noticia.split())

def stopwords(nuevo_corpus, stopw, lemma):
	for x in nuevo_corpus:
		c_norm=[]
		doc= nlp(x)
		for token in doc:
			if token.tag_ not in stopw:
				c_norm.append(token.lemma_)
		lemma.append(c_norm)

def lematizar(nuevo_corpus, lemma):
	for x in nuevo_corpus:
		c_lemma=[]
		doc= nlp(x)
		for token in doc:
			c_lemma.append(token.lemma_)
		lemma.append(c_lemma)
"""
