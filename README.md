# 📄 QueryNest – Document AI Assistant

QueryNest is a **Flask-based web application** that allows users to upload documents (PDF, Word, Excel) and interact with them using **AI-powered Question Answering (QA)**.  
The system leverages **Groq LLM** and **vector embeddings (via LlamaIndex + HuggingFace Sentence-Transformers)** to extract **context-aware, semantic answers** directly from uploaded files.

---

## ✨ Features

- 📂 **Document Upload**: Upload PDF, Word (.docx), and Excel (.xlsx) files.  
- 🤖 **LLM-Powered QA**: Uses **Groq API (Mixtral model)** for intelligent, context-driven answers.  
- 🔍 **Semantic Search**: Employs **LlamaIndex + HuggingFace embeddings** for efficient document indexing & retrieval.  
- 🌐 **Interactive UI**: Drag-and-drop upload interface with smooth user experience.  
- 📱 **Responsive Design**: Works seamlessly across desktop and mobile devices.  
- 📑 **Multi-format Support**: Process text-rich files across multiple formats in one unified system.  

---

## 🛠️ Tech Stack

### Backend
- Flask (Python)  
- LlamaIndex for semantic document indexing  
- HuggingFace Sentence-Transformers for embeddings  

### Frontend
- HTML, JavaScript, Tailwind CSS  

### Document Processing
- PyPDF → PDF Parsing  
- python-docx → Word documents  
- pandas + openpyxl → Excel files  

### AI / ML
- Groq API (Mixtral model)  

---

## 🚀 Setup Instructions

### Prerequisites
- Python **3.9+**  
- Groq API Key  

### Installation

Clone the repository:
```bash
git clone https://github.com/Aritrraa/Query-Nest---Document-AI-Assistant.git
cd Query-Nest---Document-AI-Assistant
```

Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env.local` file in the root folder with your API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:
```bash
python app.py
```

Open in browser:
```
http://127.0.0.1:5000
```

---

## 🪴 Usage

### Upload a Document  
- Drag & drop or click to upload **PDF, Word, or Excel** files.  
- The file is processed and indexed for semantic search.  

### Ask Questions  
- Enter a query in the input box.  
- QueryNest finds the most relevant context and generates an answer using **Groq LLM**.  

### Generate Summaries  
- QueryNest can also create summaries of uploaded documents for quick insights.  

---

## 📂 Project Structure

```graphql
Query-Nest---Document-AI-Assistant/
│── app.py              # Main Flask app
│── templates/
│   └── index.html      # Frontend HTML template
│── uploads/            # Stores uploaded documents
│── requirements.txt    # Python dependencies
└── .env.local          # API key config
```

---

## 📋 Requirements

Main dependencies:
- flask  
- werkzeug  
- python-dotenv  
- groq  
- llama-index  
- sentence-transformers  
- pypdf  
- python-docx  
- pandas, openpyxl  

*(See requirements.txt for the full list)*

---

## 🖼️ Screenshots / Project Images

### 1️⃣ Project Overview  
![Project Overview](project_overview1.png)
![Project Overview](project_overview2.png)
![Project Overview](project_overview3.png)

### 2️⃣ Document Upload  
![Upload Document](upload.png)

### 3️⃣ Processing & Indexing  
![Processing Document](processing.png)

### 4️⃣ Ask Questions (QA Interface)  
![QA Demo](query.png)

### 5️⃣ Document Summary Generation  
![Summary Output](summary.png)

---

## ⚡ Limitations

- Free hosting plans (e.g., Netlify/Render) have **512MB RAM limit** → large LLMs may fail.  
- Processing **large documents** takes longer to embed & query.  
- Answer quality depends on **document clarity + LLM capability**.  

---

## 🔮 Future Improvements

- Support for multiple LLMs (Gemini, OpenAI GPT, etc.).  
- Add **chat history** for multi-turn Q&A.  
- Implement **user authentication & document history**.  
- Markdown / Rich text formatting for responses.  
- Optimize for **larger documents & faster indexing**.  

---


## 🙏 Acknowledgements

- Groq API  
- LlamaIndex  
- HuggingFace Transformers  
- Tailwind CSS  

---

---

## 👨‍💻 Author

**Aritra Das**  
🚀 Developer of QueryNest · Document AI Assistant  

- 🌐 [GitHub](https://github.com/Aritrraa)  
- 💼 [LinkedIn](https://www.linkedin.com/in/aritra-das-6b5b89231/)  
