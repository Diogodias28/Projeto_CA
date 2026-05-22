import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

print("A carregar os dados...")
df = pd.read_csv("dataset_plutchik.csv")

emocoes = ['Joy', 'Sadness', 'Anger', 'Fear', 'Trust', 'Disgust', 'Surprise', 'Anticipation']
emocao_para_id = {emocao: i for i, emocao in enumerate(emocoes)}
id_para_emocao = {i: emocao for i, emocao in enumerate(emocoes)}

df['label'] = df['Plutchik_Emotion'].map(emocao_para_id)

hf_dataset = Dataset.from_pandas(df[['text', 'label']])

dataset_dividido = hf_dataset.train_test_split(test_size=0.2)
train_dataset = dataset_dividido['train']
test_dataset = dataset_dividido['test']

print("A carregar o Tokenizer e o Modelo DistilBERT...")

Notify = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(Notify)

def processar_dados(exemplos):
    return tokenizer(exemplos['text'], padding="max_length", truncation=True, max_length=64)

train_dataset = train_dataset.map(processar_dados, batched=True)
test_dataset = test_dataset.map(processar_dados, batched=True)

Notify = AutoModelForSequenceClassification.from_pretrained(Notify, num_labels=8)

argumentos_treino = TrainingArguments(
    output_dir="./resultados",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=Notify,
    args=argumentos_treino,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("A iniciar o treino...")
trainer.train()

print("Treino concluído! A guardar o modelo...")
Notify.save_pretrained("./modelo")
tokenizer.save_pretrained("./modelo")

print("O modelo está guardado!")