print("====== NORMALIZACIÓN DE TEXTO (by Billy) ===========")
#Importamos la biblioteca spacy
import spacy
import pandas as pd
import numpy as np
#conda install -c conda-forge spacy-model-es_core_news_sm
#python -m spacy download es_core_news_sm

print("Cargando dataset...")
#~ El dataset estará codificado como latin1 para que el texto sea leíble
df= pd.read_csv('corpus_noticias.csv', encoding='latin1', sep=',', engine='python')
#~ Rescatando columnas de interés
titulos=df['titular'].values
contenido= df['noticia'].values
ids= df['id'].values

print("Preprocesando texto para obtener sus tokens...")
tokens_titulos=[]
tokens_content=[]
#~ Obtención de tokens en sus respectivas listas
for seccion in titulos:
	tokens_titulos.append(seccion.split())

for noticia in contenido:
	tokens_content.append(noticia.split())

#~ Conversión de las listas a string para que las comas pertenecientes
# puramente al texto no se vean afectados para el siguiente paso
tkn_t=[]
tkn_c=[]

for t in tokens_titulos:
    tkn_t.append(str(t))

for c in tokens_content:
    tkn_c.append(str(c))

#~ Arreglo 2D con los valores de las listas de tipo objeto, generando su transpuesta
# para que quede en forma de columnas

corpus= np.array([ids, tkn_t, tkn_c], dtype=object).T
#corpus= np.array([ids, tkn_t], dtype=object).T

print("Creando nuevo corpus con los tokens...")
#~ Guarda el arreglo anterior en un dataframe de pandas con las columnas nombradas
datos= pd.DataFrame(corpus, columns=['ID', 'Titulos', 'Contenido'])
#datos= pd.DataFrame(corpus, columns=['ID', 'Titulos'])

#~ Guarda el Dataframe en un CSV, con el parámetro del índice desactivado (número de fila)
# y la codificación en latin1 para que el texto tengo un formato leíble
datos.to_csv('corpus_tokens_titulos_noticias.csv', index=False, encoding="latin1")
print("Documento CSV de los tokens creado exitosamente.\n")

print("Iniciando proceso lematización del texto...")

#~ Artículos, preposiciones, conjunciones y pronombes que se buscará eliminar
eliminar= ['PRON', 'ADP', 'DET', 'CCONJ']

tag_titulos=[]
tag_contenido=[]

nlp= spacy.load('es_core_news_sm')
print("Lematizando corpus...")
for t in titulos:
	doc1= nlp(t)
	for tag in doc1:
		#print(tag)
		tag_titulos.append((tag.text, tag.lemma_, tag.tag_))
		#	txt_titulos.append(tag.text)
		#	lem_titulos.append(tag.lemma_)
#print(tag_titulos[3][2])
print("Lematización de los TÍTULOS completada")

for c in contenido:
	doc1= nlp(c)
	for tag in doc1:
		tag_contenido.append((tag.text, tag.lemma_, tag.tag_))
print("Lematización de las NOTICIAS completada")

print("=== Corpus lematizado correctamente ===")
t_lemma=[]
txt_titulos=[]
pos_titulos=[]

c_lemma=[]
txt_cont=[]
pos_cont=[]

print("\nEliminando artículos, preposiciones, conjunciones y pronombres del corpus ('DET', 'ADP', 'CCONJ', 'PRON')...")
for x in tag_titulos:
	if x[2] not in eliminar:
		t_lemma.append(x[2])
		txt_titulos.append(x[0])
		pos_titulos.append(x[1])
print("Elementos eliminados de los TÍTULOS")

for x in tag_contenido:
	if x[2] not in eliminar:
		c_lemma.append(x[2])
		txt_cont.append(x[0])
		pos_cont.append(x[1])
print("Elementos eliminados de las NOTICIAS")

print("\n=== Stop Words eliminadas correctamente ===")

#~ Almacena las listas en un arreglo 2D en forma de columnas
t_lem_tokens= np.column_stack([txt_titulos, pos_titulos, t_lemma])
c_lem_tokens= np.column_stack([txt_cont, pos_cont, c_lemma])
#print(lem_tokens)

print("Generando dataframes con los tokens lematizados...")
#~ Guarda el arreglo anterior en un dataframe de pandas con las columnas nombradas
#datos= pd.DataFrame(lem_tokens, columns=['ID', 'Token_titulo y etiqueta', 'Token_Contenido y etiqueta'])
t_lem_tokens= pd.DataFrame(t_lem_tokens, columns=['Token_titulo', 'Token_lematizado', 'Etiqueta'])
c_lem_tokens= pd.DataFrame(c_lem_tokens, columns=['Token_noticia', 'Token_lematizado', 'Etiqueta'])

#~ Guarda el Dataframe en un CSV, con el parámetro del índice desactivado (número de fila)
# y la codificación en latin1 para que el texto tengo un formato leíble
t_lem_tokens.to_csv('tokens_titulos_lematizados.csv', sep=",", index=False, encoding="latin1")
c_lem_tokens.to_csv('tokens_noticias_lematizados.csv', sep=",", index=False, encoding="latin1")
print("Archivo CSV con los tokens lematizados creado exitosamente.\n")

print("====== FIN DE LA NORMALIZACIÓN DEL TEXTO ======")

#----------------------------------
# SECCIÓN DE CÓDIGO DESCARTADO
#----------------------------------

"""
#"Eliminar" las comas que separa cada valor del arreglo
new_titulos=[]
new_cont=[]

for t in tk_t:
    new_titulos.append(''.join(str(sec) for sec in t))

for c in tk_c:
    new_cont.append(''.join(str(cont) for cont in c))
"""

#corpus = ' '.join(str(elem) for elem in datos)
#datos= datos.astype(str)
"""
tokens=[]
for s in corpus:
    tokens.append(re.sub(r'[.]{1} [.]{1}', "", s))
"""
#print(corpus[0])

#print(datos)

#print(tokens_content)
#datos=[[],[],[]]
#datos=np.array(ids,tokens_titulos,tokens_content)
#print(datos)
        
#datos[0]= ids
#datos[1]= tokens_titulos
#print(datos)
#datos[2]= tokens_content
#print(datos[2])
#datos=np.column_stack((datos))    

#np.savetxt("tokens_titulos_noticias.csv", corpus, delimiter=",", quotechar=comillas, fmt="%s", header="ID, Titulos, Noticias", comments="")

"""
eliminar = ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 
            'El', 'La', 'Los', 'Las', 'Un', 'una', 'unos', 'Unas', 'A', 'Ante', 'Bajo', 'Cabe', 'Con', 'Contra', 'De',
		 'desde', 'en', 'entre', 'hacia', 'hasta', 'para', 'por', 'según', 'sin', 'sobre', 'tras', 'y', 'e', 'ni', 'o', 'u',
		 'Desde', 'En', 'Entre', 'Hacia', 'Hasta', 'Para', 'Por', 'Según', 'Sin', 'Sobre', 'Tras', 'Y', 'E', 'Ni', 'O', 'U',
		'pero', 'mas', 'sino', 'aunque', 'si', 'como', 'que', 'cuando', 'donde', 'porque', 'pues', 'ya', 'todavía', 'aun', 
		'Pero', 'Mas', 'Sino', 'Aunque', 'Si', 'Como', 'Que', 'Cuando', 'Donde', 'Porque', 'Pues', 'Ya', 'Todavía', 'Aun',
		'Además', 'Así',	'Entonces', 'Luego', 'Conque', 'Mientras', 'Como', 'Aunque', 'Yo', 'Tú', 'Él', 'Ella', 'Usted', 
		'además', 'así',	'entonces', 'luego', 'conque', 'mientras', 'como', 'aunque', 'yo', 'tú', 'él', 'ella', 'usted', 
		'nosotros', 'nosotras', 'vosotros', 'vosotras', 'ellos', 'ellas', 'ustedes', 'mí', 'ti', 'conmigo', 'contigo',
		'Nosotros', 'Nosotras', 'Vosotros', 'Vosotras', 'Ellos', 'Ellas', 'Ustedes', 'Mí', 'Conmigo', 'Contigo', 
		'sino', 'nos', 'vos', 'consigo', 'mi', 'mis', 'tu', 'tus', 'su', 'sus', 'nuestro', 'nuestra', 'nuestros',
		'Sino', 'Nos', 'Vos', 'Consigo', 'Mi', 'Mis', 'Tu', 'Tus', 'Su', 'Sus', 'Nuestro', 'Nuestra', 'Nuestros',
		'Nuestras', 'Vuestro', 'Vuestra', 'Vuestros', 'Vuestras', 'Suyo', 'Suya', 'Suyos', 'Suyas',
		'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'suyo', 'suya', 'suyos', 'suyas']
"""

"""
c_label=[]
for x in tokens_content:
	for y in x:
		#print('z= ',z)
		if y not in eliminar:
			#print('y=',y)
			c_label.append(y)
#print(c_label)
"""

"""
for c in c_label:
	#print(c)
	doc2= nlp(c)
	for tag in doc2.ents:
		cat_contenido.append((tag.text, tag.label_))
print(cat_contenido)
"""