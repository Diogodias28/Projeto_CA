import pandas as pd
from datasets import load_dataset

print("A descarregar o dataset GoEmotions...")
dataset = load_dataset("go_emotions", "simplified")

df = pd.DataFrame(dataset['train'])

# O GoEmotions tem 28 emoções (números de 0 a 27). 
# Criámos um dicionário para as mapear para as 8 de Plutchik.
mapa_plutchik = {
    # Alegria (Joy)
    17: 'Joy', 1: 'Joy', 2: 'Joy', 15: 'Joy', 20: 'Joy',
    # Tristeza (Sadness)
    25: 'Sadness', 9: 'Sadness', 14: 'Sadness', 16: 'Sadness',
    # Raiva (Anger)
    3: 'Anger', 2: 'Anger', 10: 'Anger',
    # Medo (Fear)
    19: 'Fear', 24: 'Fear',
    # Confiança (Trust)
    0: 'Trust', 18: 'Trust', 21: 'Trust', 23: 'Trust',
    # Aversão (Disgust)
    11: 'Disgust',
    # Surpresa (Surprise)
    26: 'Surprise', 7: 'Surprise',
    # Antecipação (Anticipation)
    8: 'Anticipation', 22: 'Anticipation'
}

def obter_emocao_plutchik(lista_labels):
    label_original = lista_labels[0]
    return mapa_plutchik.get(label_original, 'Neutral')

df['Plutchik_Emotion'] = df['labels'].apply(obter_emocao_plutchik)

df_final = df[df['Plutchik_Emotion'] != 'Neutral'].copy()

print(f"Temos {len(df_final)} frases mapeadas para Plutchik.")
print(df_final[['text', 'Plutchik_Emotion']].head(10))

df_final.to_csv("dataset_plutchik.csv", index=False)