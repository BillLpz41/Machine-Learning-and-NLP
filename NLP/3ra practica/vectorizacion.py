from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

corpus = ['El niño corre velozmente por el camino a gran velocidad .',
          'El coche rojo del niño es grande .',
          'El coche tiene un color rojo brillante y tiene llantas nuevas .',
          '¿ Las nuevas canicas del niño son color rojo ?'
]

# Representación vectorial binarizada
# ~ vectorizador_binario = CountVectorizer(binary=True)
vectorizador_binario = CountVectorizer(binary=True, token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
X = vectorizador_binario.fit_transform(corpus)
print (vectorizador_binario.get_feature_names_out())
# ~ print (X)#sparse matrix
# ~ print (type(X))#sparse matrix
# ~ print (type(X.toarray()))#dense ndarray
# ~ print (X.toarray())#dense ndarray
# ~ print ('Representación vectorial binarizada')
# ~ print (X.toarray())#dense ndarray

#Representación vectorial por frecuencia
vectorizador_frecuencia = CountVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
X = vectorizador_frecuencia.fit_transform(corpus)
# ~ print('Representación vectorial por frecuencia')
# ~ print (X.toarray())

#Representación vectorial tf-idf
vectorizador_tfidf = TfidfVectorizer(token_pattern= r'(?u)\w\w+|\w\w+\n|\.')
X = vectorizador_tfidf.fit_transform(corpus)
print ('Representación vectorial tf-idf')
print (X.toarray())
