import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from datasets import Dataset

print("1. A carregar o dataset de teste e o modelo...")
df = pd.read_csv("dataset_plutchik.csv")

emocoes = ['Joy', 'Sadness', 'Anger', 'Fear', 'Trust', 'Disgust', 'Surprise', 'Anticipation']
emocao_para_id = {emocao: i for i, emocao in enumerate(emocoes)}
df['label'] = df['Plutchik_Emotion'].map(emocao_para_id)

# Usar o mesmo split de 20% do treino para validação
hf_dataset = Dataset.from_pandas(df[['text', 'label']])
dataset_dividido = hf_dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset_dividido['test']

# Carregar o teu modelo guardado localmente
# Certifica-te de que o caminho aponta para a pasta onde guardaste o modelo final
caminho_modelo = "./modelo" 
tokenizer = AutoTokenizer.from_pretrained(caminho_modelo)
model = AutoModelForSequenceClassification.from_pretrained(caminho_modelo)

print("2. A tokenizar os dados de teste...")
def processar(exemplos):
    return tokenizer(exemplos['text'], padding="max_length", truncation=True, max_length=64)

test_dataset = test_dataset.map(processar, batched=True)

print("3. A realizar inferências (Previsões)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

previsoes = []
reais = []

for ex in test_dataset:
    inputs = tokenizer(ex['text'], return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    previsoes.append(pred_id)
    reais.append(ex['label'])

# --- 4. GERAR MÉTRICAS ---
acc = accuracy_score(reais, previsoes)
report = classification_report(reais, previsoes, target_names=emocoes, output_dict=True)

print(f"\n======================================")
print(f"✅ EXATIDÃO GLOBAL (ACCURACY): {acc:.4f}")
print(f"======================================\n")

df_report = pd.DataFrame(report).transpose()
print(df_report[['precision', 'recall', 'f1-score']].head(8))

# --- 5. GERAR GRÁFICO PARA O RELATÓRIO ---
f1_scores = [report[emocao]['f1-score'] for emocao in emocoes]

plt.figure(figsize=(10, 5))
plt.bar(emocoes, f1_scores, color=['#FF4B4B', '#1E90FF', '#FF8C00', '#8A2BE2', '#32CD32', '#708090', '#FFD700', '#00CED1'])
plt.title("Desempenho do Modelo por Emoção (F1-Score)")
plt.xlabel("Categorias de Plutchik")
plt.ylabel("F1-Score")
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("desempenho_emocoes.png", dpi=300, bbox_inches='tight')
print("\n[📊] Gráfico guardado com sucesso como 'desempenho_emocoes.png'!")