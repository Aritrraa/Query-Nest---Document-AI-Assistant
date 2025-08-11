import os
import tempfile
import logging
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

import pandas as pd
import docx
import pypdf

# LLM / vector imports (lazy load embeddings)
from groq import Groq
from llama_index.core import VectorStoreIndex, Document

# -------------------------
# Config & Logging
# -------------------------
load_dotenv(".env.local")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env.local")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Flask App Setup
# -------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['UPLOAD_FOLDER'] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

groq_client = Groq(api_key=GROQ_API_KEY)
_embed_model = None  # will load when needed


def get_embed_model():
    """Lazy load embedding model."""
    global _embed_model
    if _embed_model is None:
        logger.info("Loading embedding model...")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Embedding model loaded.")
    return _embed_model


# -------------------------
# File Processing
# -------------------------
def process_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                return "\n".join([p.extract_text() or "" for p in reader.pages])
        elif ext == ".docx":
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text])
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
            return df.to_string(index=False)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        raise RuntimeError(f"Error processing {file_path}: {e}")


# -------------------------
# Groq Query
# -------------------------
def query_groq(context: str, question: str, max_context_chars: int = 60000) -> str:
    if len(context) > max_context_chars:
        logger.info("Context too long, truncating...")
        context = context[-max_context_chars:]

    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
    )
    return completion.choices[0].message.content.strip()


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No file uploaded"}), 400

    combined_text_pieces = []
    filenames = []

    for file in files:
        filename = secure_filename(file.filename)
        if not filename:
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".pdf", ".docx", ".xlsx", ".xls"]:
            return jsonify({"error": f"Unsupported file: {ext}"}), 400

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        filenames.append(filename)

        try:
            text = process_file(save_path)
            if text:
                combined_text_pieces.append(text)
        except Exception as e:
            logger.exception("Error processing file: %s", filename)
            return jsonify({"error": str(e)}), 500

    if not combined_text_pieces:
        return jsonify({"error": "No text extracted"}), 400

    combined_text = "\n\n".join(combined_text_pieces)

    temp_text_fd, temp_text_path = tempfile.mkstemp(
        prefix="combined_", suffix=".txt"
    )
    os.close(temp_text_fd)
    with open(temp_text_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    session["filenames"] = filenames
    session["text_path"] = temp_text_path

    try:
        documents = [Document(text=combined_text)]
        index = VectorStoreIndex.from_documents(documents, embed_model=get_embed_model())

        temp_index_dir = tempfile.mkdtemp(prefix="llama_index_")
        index.storage_context.persist(persist_dir=temp_index_dir)
        session["index_path"] = temp_index_dir
    except Exception as e:
        logger.exception("Failed to build index")
        return jsonify({"error": f"Failed to create index: {e}"}), 500

    return jsonify({"message": "Files uploaded & processed"})


@app.route("/query", methods=["POST"])
def query_document():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    text_path = session.get("text_path")
    if not text_path or not os.path.exists(text_path):
        return jsonify({"error": "No document uploaded"}), 400

    with open(text_path, "r", encoding="utf-8") as f:
        combined_text = f.read()

    try:
        answer = query_groq(combined_text, question)
        return jsonify({"response": answer})
    except Exception as e:
        logger.exception("Groq query failed")
        return jsonify({"error": str(e)}), 500


@app.route("/summary", methods=["GET"])
def summarize_document():
    text_path = session.get("text_path")
    if not text_path or not os.path.exists(text_path):
        return jsonify({"error": "No document uploaded"}), 400

    with open(text_path, "r", encoding="utf-8") as f:
        combined_text = f.read()

    summary_prompt = (
        "Please summarize the following document into concise bullet points:\n\n"
        f"{combined_text}"
    )

    try:
        summary = query_groq(combined_text, summary_prompt)
        return jsonify({"summary": summary})
    except Exception as e:
        logger.exception("Groq summary failed")
        return jsonify({"error": str(e)}), 500


@app.route("/delete", methods=["POST"])
def delete_files():
    for filename in session.get("filenames", []):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    for path_key in ["text_path", "index_path"]:
        path = session.get(path_key)
        if path and os.path.exists(path):
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(path)
            else:
                os.remove(path)

    session.clear()
    return jsonify({"message": "All uploaded files deleted"})


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
