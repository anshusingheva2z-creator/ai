# EVA2Z Assistant

An AI-powered customer support chatbot for **EVA2Z GPS Trackers**, built with FastAPI, FAISS vector search, and Groq LLM. It answers customer questions based on a local FAQ file and supports both **English** and **Hindi**.

---

## Features

- Answers questions from a local `faq.txt` knowledge base
- Powered by **Groq LLM (LLaMA 3.3 70B)** for natural, intelligent responses
- **FAISS vector search** for semantic similarity matching
- Supports **English** and **Hindi** languages only
- Rejects questions in any other language with a polite bilingual message
- Handles broken or informal English (e.g. "gps battery how long work?")
- Smart fallback: unknown questions get a helpful redirect message instead of a wrong answer
- GPS installation video shortcut built in
- Chat UI with quick question buttons, save chat, and clear chat
- Typing indicator while the bot is processing

---

## Supported Languages

| Language | Example Question | Reply Language |
|----------|-----------------|----------------|
| English | "Is GPS safe for my warranty?" | English |
| Hindi (Devanagari) | "क्या GPS वारंटी के लिए सुरक्षित है?" | Hindi |
| Any other language | "Hola, cómo estás?" | Rejection message |

> **Note:** Only English and Hindi (Devanagari script) are supported. Questions in any other language will receive the message: *"I'm sorry, I only support English and Hindi."*

---

## Project Structure

```
EVA2Z Assistant/
|
+-- app/
|   +-- main.py        <- Full application (FastAPI server + chat UI)
|   +-- faq.txt        <- Knowledge base (Q&A pairs)
|
+-- faiss_index/       <- Auto-generated on first run (do not delete)
|   +-- index.faiss
|   +-- index.pkl
|
+-- requirements.txt
+-- README.md
```

---

## Requirements

- Python 3.10 or higher
- A **Groq API key** — get one free at https://console.groq.com

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/eva2z-assistant.git
cd eva2z-assistant
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set your Groq API key**

```bash
# Windows - Command Prompt
set GROQ_API_KEY=your_groq_api_key_here

# Windows - PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"

# macOS / Linux
export GROQ_API_KEY=your_groq_api_key_here
```

**5. Run the server**

```bash
uvicorn app.main:app --reload
```

**6. Open in your browser**

```
http://127.0.0.1:8000
```

---

## requirements.txt

```
fastapi
uvicorn
pydantic
langchain
langchain-community
faiss-cpu
sentence-transformers
deep-translator
langdetect
requests
```

---

## How It Works

```
User types a question
         |
         v
Detect language
         |
         +-- Unsupported language ---------> Reject with polite message
         |
         +-- Hindi (Devanagari) -----------> Translate to English
         |
         +-- English (standard/broken) ----> Use as-is
         |
         v
FAISS semantic search on faq.txt
         |
         +-- No close match (score too high) -> Return fallback message
         |
         v
Groq LLM reads FAQ chunk + user question -> Generates natural answer
         |
         v
Translate answer back to user's language (English -> English, Hindi -> Hindi)
         |
         v
Return answer to user
```

---

## faq.txt Format

The knowledge base must follow this exact format. Each Q&A block must be separated by a blank line:

```
Q1. Is this GPS device safe for my vehicle warranty?
A. Yes, 100% safe. No wire cutting or splicing is required.

Q2. Do you provide free installation?
A. Yes. We provide free on-site installation by trained technicians across India.

Q3. Can I install the GPS device myself?
A. Yes, if you are familiar with vehicle wiring you can self-install.
```

Rules:
- Each question starts with `Q` followed by a number and a period — e.g. `Q1.`
- Each answer starts with `A.`
- A blank line must separate every Q&A block
- Do not skip question numbers

---

## Updating the Knowledge Base

1. Edit `faq.txt` — add, update, or remove Q&A entries
2. Delete the `faiss_index/` folder so the index is rebuilt
3. Restart the server

```bash
# Delete old index - Windows
rmdir /s /q app\faiss_index

# Delete old index - macOS / Linux
rm -rf app/faiss_index

# Restart
uvicorn app.main:app --reload
```

---

## Configuration

One setting inside `main.py` can be tuned without touching anything else:

| Setting | Default | Description |
|---------|---------|-------------|
| `SIMILARITY_THRESHOLD` | `1.2` | Controls how closely a question must match an FAQ entry. Raise to `1.4` to accept weaker matches. Lower to `0.8` to be stricter. |

---

## Chat UI Features

| Feature | How to Use |
|---------|-----------|
| Quick questions | Click the blue pill buttons above the input box |
| Ask in Hindi | Type in Hindi script — reply will also be in Hindi |
| Ask in English | Type normally — reply will be in English |
| Save chat | Click the Save button in the chat header |
| Clear chat | Click the Clear button in the chat header |
| Installation video | Ask "how to install GPS" to get a direct video link |

---

## API Reference

The server exposes one REST endpoint that can be called from any external system:

**POST** `/ask`

Request body:
```json
{
  "question": "Do you provide free installation?"
}
```

Response:
```json
{
  "answer": "Yes. We provide free on-site installation by trained technicians across India."
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "System not ready" on startup | Make sure `faq.txt` is inside the `app/` folder |
| Bot gives wrong or unrelated answers | Delete the `faiss_index/` folder and restart the server |
| Groq not responding | Check that `GROQ_API_KEY` environment variable is set |
| `UnicodeEncodeError` on the home page | Make sure `main.py` is saved with UTF-8 encoding |
| Port already in use | Use a different port: `uvicorn app.main:app --port 8001` |
| Hindi reply coming in English | The question may have been detected as English — retype in clear Devanagari script |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI |
| Vector search | FAISS via LangChain |
| Embeddings model | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Translation | Google Translator via deep-translator |
| Language detection | langdetect |
| Frontend | Vanilla HTML, CSS, JavaScript (served by FastAPI) |

---

## License

This project is proprietary and built for EVA2Z internal use.
