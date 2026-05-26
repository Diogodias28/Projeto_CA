#set page(
  paper: "a4",
  margin: 2.5cm,
)

#set text(
  font: "Libertinus Serif",
  size: 11pt,
)

#set heading(numbering: "1.")

#align(center)[
  #text(18pt, weight: "bold")[Relatório do Projeto CA]
  \
  #text(13pt)[Notify: sistema de recomendação musical baseado em emoções]
  \
  #text(11pt)[Nome: _o teu nome_]
  \
  #text(11pt)[Unidade Curricular: _preencher_]
  \
  #text(11pt)[Data: _preencher_]
]

#pagebreak()
#outline()

= Resumo

Este projeto teve como objetivo desenvolver uma aplicação capaz de detetar a emoção associada a um texto introduzido pelo utilizador e, com base nessa emoção, sugerir ou criar uma experiência musical adequada no Spotify. A solução integra técnicas de processamento de linguagem natural, aprendizagem automática e integração com uma API externa, reunindo num único sistema a classificação de emoções, a geração de recomendações musicais e a personalização da interface.

= Introdução

A relação entre emoção e música é particularmente relevante em sistemas de recomendação, uma vez que a escolha musical é frequentemente influenciada pelo estado emocional do utilizador. Em vez de recomendar músicas apenas com base em popularidade ou género, este projeto procurou explorar uma abordagem mais personalizada, centrada na emoção expressa num texto.

= Tecnologias Utilizadas

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Pandas
- Datasets
- Spotipy
- Spotify API
- Ollama (servidor local de modelos LLM)
- Typst

= Arquitetura do Sistema

A solução foi organizada em três componentes principais: preparação de dados, treino do modelo e aplicação final. A integração com Ollama adiciona uma camada de geração textual responsável por respostas empáticas e breves que acompanham a criação de playlists.

= Integração com Ollama

O projeto suporta a utilização de um servidor Ollama local para gerar respostas empáticas curtas que acompanham a criação de playlists. A aplicação (`App.py`) comunica com Ollama via HTTP (endpoints expostos em `http://localhost:11434`).

Principais chamadas usadas pela aplicação:

- `GET /api/tags` — listagem de modelos disponíveis (usada para popular o seletor de modelos na sidebar).
- `POST /api/generate` — geração de texto a partir de um prompt (usada para produzir a resposta empática).

Exemplo de fluxo de integração (resumo):

- O servidor Ollama é iniciado localmente (`ollama serve`).
- É descarregado um modelo compatível (por exemplo `ollama pull <model>`).
- A aplicação verifica os modelos disponíveis (`/api/tags`) e permite ao utilizador escolher um modelo.
- Ao receber o texto do utilizador, `App.py` formata um prompt few-shot e envia-o para `/api/generate`.
- A resposta é incorporada no histórico do chat e mostrada ao utilizador.

Instalação e execução (exemplos práticos)

1) Instalar o Ollama (seguir instruções oficiais em https://ollama.com). Exemplos:

```bash
# macOS (Homebrew)
brew install ollama

# Windows (usar o instalador disponível no site)
```

2) Descarregar um modelo local (exemplo genérico):

```bash
ollama pull <nome-do-modelo>
# Exemplo: ollama pull llama2
```

3) Arrancar o servidor Ollama (porta por defeito 11434):

```bash
ollama serve
```

4) Verificar modelos disponíveis via CLI ou HTTP:

```bash
ollama list
curl http://localhost:11434/api/tags
```

5) Exemplo de requisição HTTP (curl) ao endpoint de geração:

```bash
curl -X POST "http://localhost:11434/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"model":"<nome-do-modelo>", "prompt":"Hello", "stream": false }'
```

Snippet Python mínimo (exemplo usado e adaptado em `App.py`):

```python
import requests

def ollama_generate(prompt, model='llama2', timeout=20):
    resp = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()

# Uso:
# body = ollama_generate("Say something empathetic.", model='llama2')
# print(body)
```

Nota sobre modelos: escolhe um modelo que caiba na tua máquina. Modelos grandes (tens de gigabytes) exigem GPU/recursos elevados; para desenvolvimento local, usa variantes pequenas/otimizadas ou um modelo que já saibas funcionar com a tua instalação do Ollama.

= Como o `App.py` usa o Ollama (resumo técnico)

- Ao carregar a página, `App.py` chama `obter_modelos_ollama()` que faz `GET http://localhost:11434/api/tags` para obter a lista de modelos.
- No diálogo, após inferir a emoção com o classificador local, a função `gerar_resposta_empatica()` constrói um prompt few-shot e faz `POST http://localhost:11434/api/generate` com `model` igual ao que o utilizador selecionou.
- Se Ollama não responder, a aplicação usa fallback textual local e prossegue com a geração da playlist.

= Instruções de Execução (atualizadas)

Passos para reproduzir a demo completa numa máquina de desenvolvimento:

1) Criar e ativar um ambiente Python (recomendado `venv` ou conda) e instalar dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows PowerShell / cmd
pip install -r requirements.txt
```

2) Preparar o modelo de emoção (se ainda não estiver):

```bash
python data_preparation.py
python model_training.py
# isto gera a pasta ./modelo com tokenizer e pesos
```

3) Instalar e arrancar o Ollama (ver secção anterior). Certifica-te que um modelo está disponível e que `ollama serve` está a correr antes de iniciar a app.

4) Lançar a aplicação Streamlit:

```bash
streamlit run App.py
```

5) No browser, escolhe géneros, escreve o teu texto e confirma que Ollama aparece no separador lateral (ou usa fallback se não estiver disponível).

= Secção: Notas sobre Reprodutibilidade e Testes

- Se estiveres a utilizar uma máquina com recursos limitados, considera não executar o treino localmente e em vez disso usar o modelo já treinado disponível em `./modelo` no repositório.
- Para verificar rapidamente se o Ollama responde, executa `curl http://localhost:11434/api/tags`.
- Guarda logs do treino e métricas (por exemplo, através do `Trainer` da Hugging Face) para completar a secção de Resultados com números reais.

= Referências

- Documentação Ollama — https://ollama.com
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Hugging Face Transformers Documentation.
- Streamlit Documentation.
- Spotify Web API Documentation.
