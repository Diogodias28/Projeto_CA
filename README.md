# Notify — Emotion-Aware Music Recommendation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/NLP-DistilBERT-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Spotify](https://img.shields.io/badge/Integration-Spotify-1DB954?logo=spotify&logoColor=white)](https://developer.spotify.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black)](https://ollama.com/)

Notify is an emotion-aware music recommendation application developed for the **Affective Computing** course of the **Master's Degree in Artificial Intelligence at the University of Minho (UMinho)**.

## Problem and solution

People often look for music that matches how they feel, but generic genre-based recommendation does not capture the emotional context of a message. Notify addresses this problem by classifying a user's text into one of eight Plutchik emotions and using that signal to generate a progressive 20-track playlist.

The application combines:

- a locally fine-tuned DistilBERT classifier for emotion detection;
- a local Spotify track dataset for deterministic, feature-based selection;
- Spotify OAuth for creating or updating playlists;
- an optional local Ollama model for concise empathetic responses.

## Features and technical architecture

### Main features

- Local account registration and login for the demo.
- Text emotion classification into `Joy`, `Sadness`, `Anger`, `Fear`, `Trust`, `Disgust`, `Surprise`, or `Anticipation`.
- Genre filtering with a local Spotify tracks dataset.
- Recommendation based on `valence`, `energy`, `popularity`, and genre.
- Gradual mood progression from the detected emotion towards a positive target mood.
- Spotify playlist creation or replacement through OAuth.
- Optional local empathetic response generation through Ollama.
- Fallback behaviour when Ollama or Spotify is unavailable.

### Architecture

1. `data_preparation.py` downloads GoEmotions through Hugging Face `datasets`, maps the original labels to the Plutchik taxonomy, and writes `dataset_plutchik.csv`.
2. `model_training.py` fine-tunes `distilbert-base-uncased` with Hugging Face Transformers and stores the model/tokenizer in `modelo/`.
3. `App.py` loads the local classifier and `dataset/spotify_musics_limpo.csv`, classifies user input, selects tracks, calls Ollama when available, and manages Spotify playlists.
4. `AvaliarModelo.py` evaluates the trained classifier against the prepared dataset.

The project is a local Streamlit application. The LLM and emotion classifier run locally; Spotify is the only external service required to create a playlist.

## Installation and execution

### Requirements

- Python 3.10 or newer.
- A Spotify Developer application if playlist creation is required.
- Ollama is optional and only required for generated empathetic responses.
- Enough disk space and memory for the Hugging Face model and datasets.

### 1. Install Python dependencies

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Configure Spotify without committing credentials

Copy `.env.example` to `.env` and fill in the values:

```powershell
Copy-Item .env.example .env
```

Set the same values in the shell before starting the app:

```powershell
$env:SPOTIFY_CLIENT_ID = "your_client_id"
$env:SPOTIFY_CLIENT_SECRET = "your_client_secret"
$env:SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8080"
```

Add `http://127.0.0.1:8080` as a Redirect URI in the Spotify Developer Dashboard. The first playlist operation opens the Spotify OAuth flow and creates `.spotify_cache`; this file must remain local.

### 3. Prepare the emotion dataset and model

Run these commands from the repository root:

```powershell
python data_preparation.py
python model_training.py
```

The first command creates `dataset_plutchik.csv`. The second downloads `distilbert-base-uncased`, fine-tunes it, and writes the resulting model to `modelo/`. Training can take several minutes and benefits from a GPU.

### 4. Provide the music dataset

Download the [Spotify Tracks Dataset from Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset), clean it if necessary, and place the resulting CSV at:

```text
dataset/spotify_musics_limpo.csv
```

The application expects a genre column (for example, `track_genre`), a Spotify track identifier, and audio/features columns including `valence`, `energy`, and `popularity`.

### 5. Optional: start Ollama

Install Ollama, then download and start a model:

```powershell
ollama pull llama3
ollama serve
```

The application uses Ollama at `http://localhost:11434`. If it is not running, Notify keeps working with a local fallback response.

### 6. Run the Streamlit application

With the virtual environment active and the required files in place:

```powershell
streamlit run App.py
```

Open the URL printed by Streamlit, sign in with a local demo account, choose genres, and submit a message. Notify detects the emotion, selects the tracks, generates the response, and creates or updates the Spotify playlist when Spotify credentials are configured.

## Attribution

This project was designed and implemented collaboratively as academic coursework for the **Affective Computing** course in the **Master's Degree in Artificial Intelligence at the University of Minho (UMinho)**.

## References

- [GoEmotions](https://huggingface.co/datasets/go_emotions)
- [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- [Spotify Web API](https://developer.spotify.com/documentation/web-api)
- [Ollama](https://ollama.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
