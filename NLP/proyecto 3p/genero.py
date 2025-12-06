import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch.optim as optim

# Cargar y preparar los datos
data = pd.read_csv("/content/Politic_Train.csv")
texts = data["tweet"].tolist()
labels = data["gender"].tolist()

# Dividir los datos en conjuntos de entrenamiento y prueba
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=0)

# Tokenizar el texto
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Codificar las etiquetas
label_map = {"female": 0, "male": 1}
train_labels_encoded = [label_map[label] for label in train_labels]
test_labels_encoded = [label_map[label] for label in test_labels]

# Crear conjuntos de datos y dataloaders
train_dataset = TensorDataset(torch.tensor(train_encodings["input_ids"]),
                              torch.tensor(train_encodings["attention_mask"]),
                              torch.tensor(train_labels_encoded))
test_dataset = TensorDataset(torch.tensor(test_encodings["input_ids"]),
                             torch.tensor(test_encodings["attention_mask"]),
                             torch.tensor(test_labels_encoded))
train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=30)

# Cargar el modelo BERT pre-entrenado
model = BertForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased", num_labels=2)

# Configurar el optimizador
optimizer = AdamW(model.parameters(), lr=1e-5)
# Configurar el optimizador SGD
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Configurar el optimizador RMSprop
#optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Ajustar los parámetros del optimizador SGD
#optimizer.param_groups[0]['lr'] = 0.001
#optimizer.param_groups[0]['momentum'] = 0.5

# Ajustar los parámetros del optimizador RMSprop
#optimizer.param_groups[0]['lr'] = 0.001
#optimizer.param_groups[0]['alpha'] = 0.9
#optimizer.param_groups[0]['eps'] = 1e-8
#optimizer.param_groups[0]['weight_decay'] = 0.001



# Entrenar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(7):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}: Average loss: {average_loss}")

# Evaluar el modelo
model.eval()
total_predicted = []
total_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs.logits, dim=1)
        predicted_labels = predicted_labels.tolist()

        total_predicted.extend(predicted_labels)
        total_labels.extend(labels.tolist())

# Obtener el informe de clasificación
report = classification_report(total_labels, total_predicted, target_names=["female", "male"])
print(report)