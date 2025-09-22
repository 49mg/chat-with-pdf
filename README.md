# 📄 Chat with PDF

A simple web application built with **Gradio** and **LangChain** that lets you upload a PDF and chat with it. The app uses a **local LLM via Ollama** to answer questions **only** from the PDF content.

---

## ✨ Features

- **Easy Interface** – Simple GUI to upload files and start chatting instantly.  
- **Runs Locally** – Uses Ollama LLMs on your own machine for **privacy and security**.  
- **Accurate Answers** – Retrieval-Augmented Generation (RAG) ensures responses stay inside the PDF context.  
- **Flexible** – Powered by LangChain for document processing and retrieval chains.  

---

## ⚙️ How It Works

1. **Load & Process PDF** – The app splits the uploaded PDF into chunks.  
2. **Create Vector Store** – Chunks are embedded with `nomic-embed-text` and stored in FAISS.  
3. **Query & Search** – Your question is embedded and matched with the most relevant chunks.  
4. **Generate Answer** – `gemma3:1b` model generates an accurate response from the retrieved context.  

---

## 📋 Prerequisites

- **Python 3.8+**  
- **uv** (fast Python package manager) → `pip install uv`  
- **Ollama** (download from official website and keep it running)  

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```
git clone <your-repository-link>
cd <your-repository-name>
```

### 2. Create & Activate Virtual Environment
```
# Create
python -m venv venv

# Activate (Windows)
.env\Scriptsctivate

# Activate (Linux / macOS)
source venv/bin/activate
```

### 3. Install Requirements
```
uv pip install -r requirements.txt
```

### 4. Download Ollama Models
Make sure Ollama is running, then:  
```
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

### 5. Run the App
```
python main.py
```
Open the **local URL** or **public URL** shown in the terminal to use the app.

---

## 🔧 Customization: Changing Models

Inside `main.py`, edit the `PdfChatbot` class:  
```python
class PdfChatbot:
    def __init__(self, model_name: str = "gemma3:1b"):
        self.model = OllamaLLM(model=model_name)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
```
Pull any new models first:  
```
ollama pull <new-model-name>
```

---

## 📝 How to Use

1. Open the link from the terminal in your browser.  
2. Click **“Upload a PDF”** and select your file.  
3. Wait for the processing confirmation.  
4. Type your question at the bottom and hit Enter.  
5. See your answer instantly in the chat window.  

---

## 🛠️ Tech Stack

- Python, Gradio, LangChain, FAISS  
- Ollama LLMs (Gemma3:1b & Nomic-embed-text)  

---

💡 **Tip:** This project is perfect for local document Q&A with full privacy.  
