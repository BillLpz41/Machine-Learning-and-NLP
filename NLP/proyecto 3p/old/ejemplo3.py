from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing
from sklearn.pipeline import Pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "piubamas/beto-contextualized-hate-speech"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

id2label = [model.config.id2label[k] for k in range(len(model.config.id2label))]

def predict(text):
    encoding = tokenizer.encode_plus(text, truncation=True, padding=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predicted_label_idx = torch.argmax(logits).item()
    predicted_label = id2label[predicted_label_idx]

    return predicted_label

# Leactura del archivo csv original
df = pd.read_csv("normalizado_development_test.csv", encoding='utf-8', sep=',', engine='python')

# Columnas con los datos a utilizar para la predicción
x = df.drop(['tweet'], axis=1).values

# Realizar la predicción para cada elemento en x y guardar los resultados en una nueva columna
predictions = []
for element in x:
    text = element[0]  # Suponiendo que el texto se encuentra en la primera columna de x
    prediction = predict(text)  # Realizar la predicción utilizando tu función predict
    predictions.append(prediction)

# Agregar los resultados al DataFrame original
df['prediction'] = predictions

# Guardar el DataFrame con los resultados en un nuevo archivo CSV
df.to_csv("predictions.csv", index=False)
