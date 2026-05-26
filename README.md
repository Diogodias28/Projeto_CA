# Projeto_CA

Sistema de recomendação musical orientado por emoções, com classificação de texto, integração com Spotify e geração de respostas empáticas através do Ollama.

## Visão geral

O projeto tem três partes principais:

1. Preparação de dados e criação do dataset de emoções em formato Plutchik.
2. Treino de um classificador de emoções com DistilBERT.
3. Aplicação final em Streamlit, com autenticação, análise emocional, integração com Spotify e resposta textual gerada pelo Ollama.

## Origem dos dados

### Dataset de emoções

O dataset usado para treinar o classificador de emoções vem do GoEmotions, através da biblioteca `datasets` da Hugging Face:

- GoEmotions: https://huggingface.co/datasets/go_emotions

O script [`data_preparation.py`](data_preparation.py) faz o download do dataset, mapeia as labels originais para 8 emoções de Plutchik e gera o ficheiro [`dataset_plutchik.csv`](dataset_plutchik.csv).

### Dataset musical

O dataset de músicas usado para selecionar as faixas da playlist vem do Kaggle:

- Spotify Tracks Dataset: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Depois de descarregado, o ficheiro limpo deve ficar em:

- [`dataset/spotify_musics_limpo.csv`](dataset/spotify_musics_limpo.csv)

Este ficheiro é o que a aplicação usa para escolher músicas com base em `valence`, `energy`, `popularity` e `track_genre`.

## Estrutura do projeto

- [`App.py`](App.py) - aplicação principal em Streamlit.
- [`data_preparation.py`](data_preparation.py) - criação do dataset de emoções em Plutchik.
- [`model_training.py`](model_training.py) - treino do modelo de classificação.
- [`requirements.txt`](requirements.txt) - dependências Python.
- [`dataset/`](dataset) - dataset musical.
- [`modelo/`](modelo) - modelo e tokenizer treinados.
- [`resultados/`](resultados) - checkpoints e outputs do treino.
- [`utilizadores.json`](utilizadores.json) - dados locais de utilizadores da demo.

## Requisitos

- Python 3.10 ou superior.
- Conta Spotify Developer para autenticação OAuth (Estamos a utilizar uma conta com Premium já para criar as playlists). 
- Ollama instalado localmente, para usar a geração empática por LLM.

## Instalação

### 1. Instalar dependências

```cmd
pip install -r requirements.txt
```

## Preparação dos dados e treino

### 1. Gerar o dataset em Plutchik

```cmd
python data_preparation.py
```

Isto cria o ficheiro [`dataset_plutchik.csv`](dataset_plutchik.csv).

### 2. Treinar o modelo de emoções

```cmd
python model_training.py
```

No fim deste passo, o modelo e o tokenizer ficam guardados na pasta [`modelo/`](modelo).

## Configuração do dataset musical

1. Descarrega o dataset original do Kaggle:
	- https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
2. Pipeline de limpeza(opcional)
3. Garante que o ficheiro final está em [`dataset/spotify_musics_limpo.csv`](dataset/spotify_musics_limpo.csv).

O script [`App.py`](App.py) procura esse ficheiro diretamente para construir as playlists.

## Configuração do Ollama

O Ollama é opcional, mas é o componente usado para gerar respostas curtas e empáticas no chat.

### 1. Instalar o Ollama

- https://ollama.com

### 2. Arrancar o servidor

```cmd
ollama serve
```

### 3. Fazer download de um modelo

```cmd
ollama pull llama3
```

### 4. Verificar os modelos disponíveis

```cmd
ollama list
```

O `App.py` consulta o Ollama em `http://localhost:11434/api/tags` e usa `POST /api/generate` para obter a resposta textual.

## Como executar a demo

1. Garantir que o dataset musical está em [`dataset/spotify_musics_limpo.csv`](dataset/spotify_musics_limpo.csv).
2. Garantir que o modelo treinado está na pasta [`modelo/`](modelo).
3. Iniciar o Ollama com `ollama serve`.
4. Abrir um terminal na raiz do projeto e correr:

```cmd
streamlit run App.py
```

5. Abrir o URL mostrado pelo Streamlit no browser.
6. Fazer login ou criar uma conta local.
7. Selecionar um ou mais géneros.
8. Escrever uma mensagem no chat.
9. A aplicação:
	- classifica a emoção com o modelo local;
	- gera uma resposta empática via Ollama, se disponível;
	- escolhe músicas do dataset;
	- cria ou atualiza uma playlist no Spotify e devolve o link.

## Reproduzir os resultados

Para reproduzir a demo e os resultados do projeto, segue esta ordem:

```cmd
python data_preparation.py
python model_training.py
ollama serve
streamlit run App.py
```

## Notas importantes

- O modelo de emoção é treinado localmente; o treino pode demorar dependendo da máquina.
- O ficheiro [`utilizadores.json`](utilizadores.json) guarda utilizadores da demo em disco.
- Se o Ollama não estiver disponível, a app continua a funcionar com uma resposta de fallback.
- A playlist pode ser criada como nova ou substituir uma playlist existente, dependendo da URL configurada na sidebar da app.

## Referências

- GoEmotions dataset: https://huggingface.co/datasets/go_emotions
- Spotify Tracks Dataset: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- Ollama: https://ollama.com
- Streamlit: https://streamlit.io
- Hugging Face Transformers: https://huggingface.co/docs/transformers