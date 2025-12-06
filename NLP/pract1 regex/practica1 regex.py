import re
import pandas as pd

df= pd.read_csv('tweets_1.csv', sep=',', engine= 'python')

tw =[]
for i in (df['Tweet']):
  tw.append(i)
print("Cantidad de tweets: ", len(tw))


#Hashtags
#tw[1]="@12345678901234567"
hstag= re.findall(r'#(\w|[\u00C0-\u017F]|\U000000F1|\U000000D1)+', str(tw))
print("Cantidad de hashtags: ", len(hstag))

#Nombres de usuarios
user=re.findall(r'@[\w]{1,15}\b', str(tw))
print("Cantidad de nombres de usuarios: ", len(user))

#Hora
#'((([0-1]?[0-9]|2[0-3]):[0-5][0-9])|((0[0-9]|1[0-2]):[0-5][0-9](\s)?[(a\.m\.)(p\.m\.)])){1}'
#((([0|1]?[0-9])|(1[0-2]?))?:([0-5][0-9])\s?((a\.m\.)|(p\.m\.))|([0|1]?[0-9]|2[0-4])?:([0-5][0-9])){1}

horas=re.findall(r'(((0?\d|1[012])( [aAPp]\.?[Mm]\.?)\b)|((0?\d|1[012])(:[0-5]\d)( [aAPp]\.?[Mm]\.?)?\b)|(([01]?\d|2[0-3]):[0-5]\d))', str(tw))
#print(horas)
print("Cantidad de horas: ", len(horas))

#Fecha
#meses=['enero', 'febrero', 'marzo', 'abril','mayo','junio','julio',
#       'agosto','septiembre', 'octubre','noviembre','diciembre']

#'(([0-3][0-1]|([0[1-9]|1[0-2]])|([1|2][\d]{3}))(/|-)?(([0[1-9]|1[0-2]])|([0-3][0-1])|([1|2][\d]{3}))(/|-)?(([1|2][\d]{3}))|([0-3][0-1])|([0[1-9]|1[0-2]]))'
#fechas= re.findall(r'(0?[1-9]|[12][0-9]|3[01])[./-](0?[1-9]|1[012])', str(tw))
#fechas=re.findall(r'(?:(?:[1-31]/[1-12]|[1-12]/[1-31])/[\d{4}]|[1-31]/[\d{4}/[1-12]|[1-12]/[\d{4}]/[1-31]|[\d{4}/(?:[1-31]/[1-12]|[1-12]/[1-31])', str(tw))

#fechas= re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)? [A-Za-z]+ \d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[/-][A-Za-z]{3}[/-]\d{2,4}\b',str(tw))
#fechas= re.findall(r'(([012]?\d)|3[01])?[-/\. ][a-zA-Z]{4,10}\b([-/\. ](\d{2,4})?\b | (([012]?\d)[-/\.](0?[1-9]|1[012])[-/\.](\d{2,4})?\b) | (()))', str(tw))
#fechas= re.findall(r'(\d{1,2}[-/\.](\d{1,2}|[a-zA-Z]{3})\b[-/\.]\d{2,4}\b)|(\d{1,2}( de )[A-Za-z]+\b( [de|del] )?\d{2,4}\b)|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[/-][A-Za-z]{3}[/-]\d{2,4}\b', str(tw))

fechas= re.findall(r'(\d{1,2}[-/\.]\d{1,2}([-/\.]\d{2,4}\b)?)|(\d{1,2}\s(d|de)\s(([eE]nero)|([fF]ebrero)|([mM]arzo)|([aA]bril)|([mM]ayo)|([jJ]unio)|([jJ]ulio)|([aA]gosto)|([sS]eptiembre)|([oO]ctubre)|([Nn]oviembre)|([Dd]iciembre))((\sde|\sdel)\d{2,4}\b)?)|(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b)|(\d{1,2}[-/\.][A-Za-z]{3}[-/\.]\d{2,4}\b)', str(tw))
#print(fechas)
print("Cantidad de fechas: ", len(fechas))

#Emoticonos
emojis=re.findall(r'[\U0001F000-\U0001FFFF]', str(tw))
#print(emojis)
print("Cantidad de emoticonos: ", len(emojis))

"""
cantEmot= 0
for i in tw:
  if(chr(e)==tw):
    cantEmot+=0
    e+=1
  else:
    e+=1

for x in emojis:
  if(x == ''):
    emojis.remove(x)
"""
#^\\U0001F(?:6(?:06|43)|923)$
#print("Cantidad de emoticonos: ", cantEmot)