import re
import pandas as pd

#Leactura del archivo csv original
df = pd.read_csv('tweets_1.csv', sep=',', engine='python')
#Columna de tweets
x = df['Tweet'].values

lista = []

usuario = re.compile("@[a-zA-Z|\d|\_]+")
hashtag = re.compile("#[a-zA-Z|\d|\_]+")
hora = re.compile("[0(\d)|1\d]\x3a56")
fecha = 0
emoticon = 0 

for linea in x:
    lista.append(linea)

texto = "19:56"
texto_lista = hora.findall(texto)
print(len(texto_lista))
print(texto_lista)

usuario_lista = usuario.findall(str(lista))
hashtag_lista = hashtag.findall(str(lista))
hora_lista = hora.findall(str(lista))

print(len(usuario_lista))
print(len(hashtag_lista))
print(len(hora_lista))