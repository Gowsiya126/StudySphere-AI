# StudySphere AI

A beginner-friendly local RAG study assistant built with Python.

## Features
- Upload one or more text-based PDFs
- Ask questions grounded in indexed document chunks
- ELI5 explanations
- MCQ and short-answer question generation
- Document summaries
- Retrieved source filename and page references

## Requirements
- Windows 10/11
- Python 3.12.x
- Ollama for local LLM responses
- Internet for the first download of Python packages, the embedding model and the Ollama model

## Setup (PowerShell)
```powershell
cd D:\Development\Projects\StudySphereAI
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Ollama from https://ollama.com, open it, then run:
```powershell
ollama pull llama3.2:3b
```

Run the app:
```powershell
python -m streamlit run app.py
```

The browser should open at http://localhost:8501.

## Important
- Upload PDFs in the left sidebar and click **Process PDFs** before asking questions.
- Scanned/image-only PDFs may not contain selectable text. Run OCR on those PDFs first.
- The first embedding-model download may take a few minutes.
- Ollama must be running locally for answer generation.
- This is a learning project. Verify important answers against the original document.
