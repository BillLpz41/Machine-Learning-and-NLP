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
#titulos=df['titular'].values
contenido= df['noticia'].values
#ids= df['id'].values

print("Preprocesando corpus de noticias para obtener sus tokens...")
tokens_content=[]
#~ Obtención de tokens en sus respectivas listas

for noticia in contenido:
	tokens_content.append(noticia.split())

#~ Eliminando las comas que separan los caracteres
tokens_string=[]

for i in tokens_content:
	tokens_string.append(' '.join(i))

#~ Arreglo 2D con los valores de las listas de tipo objeto, generando su transpuesta
# para que quede en forma de columnas

corpus= np.array(tokens_string, dtype=object).T
#corpus= np.array([ids, tkn_t], dtype=object).T

print("Creando nuevo corpus con los tokens...")
#~ Guarda el arreglo anterior en un dataframe de pandas con las columnas nombradas
datos= pd.DataFrame(corpus)
#datos= pd.DataFrame(corpus, columns=['ID', 'Titulos'])

#~ Guarda el Dataframe en un CSV, con el parámetro del índice desactivado (número de fila)
# y la codificación en latin1 para que el texto tengo un formato leíble
datos.to_csv('tokens_noticias.csv', index=False, encoding="latin1", header=False)
print("Documento CSV de los tokens creado exitosamente.\n")

print("Iniciando proceso lematización del texto...")
nlp= spacy.load('es_core_news_sm')

print("Lematizando corpus...")

txt_cont=[]
#~ x obtiene una noticia entera, doc tendrá ese mismo texto con los atributos de
# spacy. Posteriormente token obtiene una palabra del texto de doc y se comprueba si
#  su etiqueta está en la lista de elementos que se desea eliminar
for x in contenido:
	c_lemma=[]
	doc= nlp(x)
	for token in doc:
		c_lemma.append(token.lemma_)
	txt_cont.append(c_lemma)

#~ Eliminando las comas que separan los caracteres
tokens_string=[]

for i in txt_cont:
	tokens_string.append(' '.join(i))

print("=== Corpus lematizado correctamente ===\n")

#~ Almacena las listas en un arreglo 2D en forma de columnas mediante la transpuesta de la matriz
cont_lemma= np.array(tokens_string, dtype=object).T

print("Generando dataframes con los tokens lematizados...")

#~ Guarda el arreglo anterior en un dataframe de pandas con las columnas nombradas

#datos= pd.DataFrame(lem_tokens, columns=['ID', 'Token_titulo y etiqueta', 'Token_Contenido y etiqueta'])
cont_lemma= pd.DataFrame(cont_lemma)
#c_lem_tokens= pd.DataFrame(c_lem_tokens, columns=['Token_noticia', 'Token_lematizado', 'Etiqueta'])

#~ Guarda el Dataframe en un CSV, con el parámetro del índice desactivado (número de fila)
# y la codificación en latin1 para que el texto tengo un formato leíble

#np.savetxt('noticia.csv', t_lem_tokens, delimiter=",", fmt="%s", encoding="latin1", header="ID, Noticia")
cont_lemma.to_csv('noticia_lematizado.csv', index=False, encoding="latin1", header=False)
#c_lem_tokens.to_csv('tokens_noticias_lematizados.csv', sep=",", index=False, encoding="latin1")

print("Fichero CSV de los tokens lematizados creado exitosamente.\n")

print("\nEliminando artículos, preposiciones, conjunciones y pronombres del corpus ('DET', 'ADP', 'CCONJ', 'PRON')...")

#~ Artículos, preposiciones, conjunciones y pronombes que se buscará eliminar
eliminar= ['PRON', 'ADP', 'DET', 'CCONJ']

norm_cont=[]
#~ x obtiene una noticia entera, doc tendrá ese mismo texto con los atributos de
# spacy. Posteriormente token obtiene una palabra del texto de doc y se comprueba si
#  su etiqueta está en la lista de elementos que se desea eliminar
for x in contenido:
	c_norm=[]
	doc= nlp(x)
	for token in doc:
		if token.tag_ not in eliminar:
			c_norm.append(token.lemma_)
	norm_cont.append(c_norm)

print("\n=== Stop Words eliminadas del corpus ===")
print("=== Corpus normalizado correctamente ===\n")

#~ Eliminando las comas que separan los caracteres
tokens_string=[]

for i in norm_cont:
	tokens_string.append(' '.join(i))

#~ Almacena las listas en un arreglo 2D en forma de columnas mediante la transpuesta de la matriz
cont_norm= np.array(tokens_string, dtype=object).T

print("Generando dataframes con los tokens lematizados...")

#~ Guarda el arreglo anterior en un dataframe de pandas con las columnas nombradas

#datos= pd.DataFrame(lem_tokens, columns=['ID', 'Token_titulo y etiqueta', 'Token_Contenido y etiqueta'])
cont_norm= pd.DataFrame(cont_norm)
#c_lem_tokens= pd.DataFrame(c_lem_tokens, columns=['Token_noticia', 'Token_lematizado', 'Etiqueta'])

#~ Guarda el Dataframe en un CSV, con el parámetro del índice desactivado (número de fila)
# y la codificación en latin1 para que el texto tengo un formato leíble

#np.savetxt('noticia.csv', t_lem_tokens, delimiter=",", fmt="%s", encoding="latin1", header="ID, Noticia")
cont_norm.to_csv('noticia_normalizado.csv', index=False, encoding="latin1", header=False)
#c_lem_tokens.to_csv('tokens_noticias_lematizados.csv', sep=",", index=False, encoding="latin1")

print("Archivo CSV con la noticias normalizadas creado exitosamente.\n")

print("====== FIN DE LA NORMALIZACIÓN DEL TEXTO ======")


#----------------------------------
# SECCIÓN DE CÓDIGO DESCARTADO
#----------------------------------

"""
#~ Conversión de las listas a string para que las comas pertenecientes
# puramente al texto no se vean afectados para el siguiente paso
tkn_c=[]

for c in tokens_content:
    tkn_c.append(str(c))
"""

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

#t_lem_tokens= np.column_stack([txt_titulos, pos_titulos, t_lemma])

#t_lem_tokens= np.array([ids, t_lemma], dtype=object)
#c_lem_tokens= np.column_stack([txt_cont, pos_cont, c_lemma])
#print(lem_tokens)


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

#print(txt_titulos)

#for t in txt_titulos:
#    lemma.append(str(t))
#lemma=np.array(lemma, dtype=object)
#print(lemma)


"""
for c in c_label:
	#print(c)
	doc2= nlp(c)
	for tag in doc2.ents:
		cat_contenido.append((tag.text, tag.label_))
print(cat_contenido)
"""
"""
for x in contenido:
	c_lemma=[]
	doc= nlp(x)
	for token in doc:
		if token.tag_ not in eliminar:
			c_lemma.append(token.lemma_)
	txt_cont.append(c_lemma)
"""
