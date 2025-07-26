# 🚀 QA Chatbot with RAG: Multilingual Magic for Bangla & English

Welcome to the **QA Chatbot with Retrieval-Augmented Generation (RAG)**, a powerful tool designed to answer questions from scanned Bangla and English PDFs with precision and flair! This project combines cutting-edge NLP, OCR, and vector search to deliver accurate and context-aware responses. Whether you're querying in Bangla or English, this chatbot has you covered! 🌟

---

## 📋 Table of Contents

- Features
- Setup Guide
- Tools & Libraries
- Deployment
- Sample Queries
- API Documentation
- Technical Insights
- Future Enhancements
- Author

---

## 🌟 Features

- 📜 **Multilingual Support**: Seamlessly handles Bangla and English queries from scanned PDFs.
- 🧠 **RAG-Powered**: Combines retrieval from Pinecone with GPT-4 for accurate, context-rich answers.
- 📸 **OCR Integration**: Extracts text from scanned Bangla PDFs using PyMuPDF and Tesseract.
- ⚡ **FastAPI Backend**: Lightning-fast API for starting chats, retrieving history, and evaluating responses.
- 📊 **Evaluation Metrics**: Measures relevance and groundedness of answers for quality assurance.
- 🌐 **Deployed & Accessible**: Live on Render with interactive API docs.

---

## 🛠️ Setup Guide

Get started in just a few steps! Follow this guide to set up the project locally.

1. **Clone the Repository**

```bash
git clone https://github.com/shakilahammed055/chatbot_using_rag.git
cd chatbot_using_rag
```

2. **Create a Virtual Environment**

```bash
python -m venv .venv
.venv\Scripts\activate  # On Linux: source .venv/bin/activate
```

3. **Install Required Libraries**

```bash
pip install -r requirements.txt
```

4. **Set Up Environment Variables**

Create a `.env` file in the project root and add your API keys:

```bash
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

5. **Run the App**

```bash
uvicorn app.main:app --reload
```

🎉 You're ready to chat with the bot!

---

## 🧰 Tools & Libraries

This project is powered by a robust stack of tools and libraries tailored for multilingual text processing and retrieval.

| **Tool/Library**                  | **Purpose**                                                        |
| --------------------------------- | ------------------------------------------------------------------ |
| **PyMuPDF (fitz)**                | Extracts images from scanned Bangla PDFs.                          |
| **pytesseract + PIL.Image**       | OCR engine for extracting Bangla text from PDF images.             |
| **unicodedata + re**              | Normalizes and cleans Bangla text for consistency.                 |
| **langchain**                     | Handles document chunking, prompting, and retrieval orchestration. |
| **langchain_openai**              | Generates embeddings using OpenAI’s `text-embedding-3-large`.      |
| **Pinecone + langchain_pinecone** | Stores and retrieves document chunks in a vector database.         |
| **FastAPI**                       | Serves API endpoints for chat, history, and evaluation.            |
| **uvicorn**                       | ASGI server for hosting the FastAPI backend.                       |
| **NumPy**                         | Performs vector math for cosine similarity in evaluation.          |
| **OpenAI API (gpt-4)**            | Powers LLM-based answer generation.                                |
| **os, hashlib, uuid**             | Manages file access, text hashing, and unique ID generation.       |

---

## 🌐 Deployment

The QA Chatbot is live and ready to use! Access it at:

👉 **Live App**: https://chatbot-using-rag.onrender.com\
📘 **API Docs**: https://chatbot-using-rag.onrender.com/docs

---

## 💬 Sample Queries

Here’s a glimpse of how the chatbot responds to queries in English and Bangla:

| **Language** | **Query**                                | **Response**                                                                          |
| ------------ | ---------------------------------------- | ------------------------------------------------------------------------------------- |
| **English**  | Who is Anupam’s legal guardian?          | According to the document, Anupam refers to his maternal uncle as his legal guardian. |
| **Bangla**   | কল্যাণীর প্রকৃত বয়স কত ছিল বিয়ের সময়? | ১৫ বছর                                                                                |
| **Bangla**   | অনুপমের ভাগ্যদেবতা কে?                   | অনুপমের মামা                                                                          |

---

## 📑 API Documentation

Explore the API endpoints to interact with the chatbot programmatically.

### 🟢 `GET /QA-chatbot/start-chat`

Initiates a new chat session and returns a unique session ID.

**Response**

```json
{
  "session_id": "vJEW5QlUQDWmb1-LGQj0Dw",
  "message": "New chat session started."
}
```

> **Note**: Call this endpoint to start a chat and store history based on the session ID.

### 🔴 `DELETE /QA-chatbot/end-chat/{session_id}`

Terminates and deletes a specific chat session.

> **Note**: Use this when the user ends the chat to clear their session data.

### 🟡 `POST /QA-chatbot/ask`

Submits a question for the chatbot to answer.

**Request**

```json
{
  "session_id": "Hyj1gdNjQ5elZzdg2HQ1Bw",
  "question": "Who is Anupam’s mentor in the story?"
}
```

**Response**

```json
{
  "session_id": "Hyj1gdNjQ5elZzdg2HQ1Bw",
  "question": "Who is Anupam’s mentor in the story?",
  "answer": "Anupam’s mentor is his maternal uncle, who guides him after his father’s death."
}
```

**Request (Bangla)**

```json
{
  "session_id": "Hyj1gdNjQ5elZzdg2HQ1Bw",
  "question": "তার ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

**Response**

```json
{
  "session_id": "Hyj1gdNjQ5elZzdg2HQ1Bw",
  "question": "তার ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শস্তুনাথবাবুকে সুপুরুষ বলা হয়েছে।"
}
```

> **Note**: Questions are stored in short-term memory (dictionary) for the same session ID.

### 🔁 `POST /QA-chatbot/history`

Retrieves the chat history for a given session.

**Query Parameter**

- `session_id`: `LoA60icTQLeuxmuMn_KY0Q`

**Response**

```json
{
  "session_id": "LoA60icTQLeuxmuMn_KY0Q",
  "history": [
    {
      "question": "কাকে অনুপমের ভাগ্যদেবতা বলে উল্লেখ করা হয়েছে?",
      "answer": "অনুপমের মামা।"
    },
    {
      "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
      "answer": "১৫ বছর"
    }
  ]
}
```

### 📊 `POST /QA-chatbot/evaluate`

Evaluates the chatbot’s responses against expected answers.

**Request**

```json
{
  "data": [
    {
      "question": "কাকে অনুপমের ভাগ্যদেবতা বলে উল্লেখ করা হয়েছে?",
      "expected_answer": "মামা"
    },
    {
      "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
      "expected_answer": "১৫ বছর"
    }
  ]
}
```

**Response**

```json
{
  "average_relevance": 0.7842567456519585,
  "average_groundedness": 0.3772497844742908,
  "results": [
    {
      "question": "কাকে অনুপমের ভাগ্যদেবতা বলে উল্লেখ করা হয়েছে?",
      "expected_answer": "মামা",
      "generated_answer": "অনুপমের মামা।",
      "relevance_score": 0.6956699414476105,
      "groundedness_score": 0.4671823197282297
    },
    {
      "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
      "expected_answer": "১৫ বছর",
      "generated_answer": "১৫ বছর",
      "relevance_score": 0.8728435498563065,
      "groundedness_score": 0.28731724922035196
    }
  ]
}
```

---

## 🔍 Technical Insights

### ❓ Text Extraction Method

**Used**: `PyMuPDF (fitz)` + `pytesseract` (with `PIL.Image`)\
**Why**: Scanned Bangla PDFs contain image-based text, requiring OCR. PyMuPDF extracts pages as images, and Tesseract converts them to text. Libraries like `pdfminer` or `PyPDF2` are unsuitable for non-digital text.

### ⚠️ Formatting Challenges

- **Issues**: Broken/joined Bangla characters, irregular spacing, and non-Bangla noise.
- **Solution**: Custom `clean_bangla_text()` function:

```python
def clean_bangla_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = unicodedata.normalize('NFKC', text)  # Normalize Unicode
    text = re.sub(r'[^\u0980-\u09FF\s.,?!ঃ]', '', text)  # Keep Bangla chars
    text = text.replace('্ ', '্')  # Fix broken conjuncts
    return text.strip()
```

### ✂️ Chunking Strategy

**Used**: `RecursiveCharacterTextSplitter` (LangChain) with Bangla-specific separators.\
**Config**:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50,
    separators=["\n", "।", ".", " "]
)
```

**Why**: Ensures coherent chunks with overlap for context retention, tailored for Bangla text.

### 🧬 Embedding Model

**Used**: `text-embedding-3-large` (OpenAI)\
**Why**: Superior performance for multilingual (Bangla-English) semantic similarity compared to `text-embedding-3-small` or `sentence-transformers/clip-ViT-B-32-multilingual-v1`.

### 🔗 Query-Chunk Comparison

**Method**: Pinecone vector database with ANN search (cosine similarity).\
**Process**:

1. Embed query using `text-embedding-3-large`.
2. Retrieve top-k chunks from Pinecone based on vector similarity.
3. Pass relevant chunks to GPT-4 for answer generation.

**Why**: Pinecone offers fast, scalable retrieval for high-dimensional embeddings.

### 🎯 Result Relevance

The system delivers relevant answers but has room for improvement:

- Refine chunking for Bangla coherence.
- Explore fine-tuned Bangla embeddings.
- Enhance OCR post-processing.
- Add query reformulation for vague inputs.
- Implement re-ranking with GPT-based scoring.

---

## 📌 Future Enhancements

- **Advanced OCR**: Use Transformer-based OCR (e.g., TrOCR) for better Bangla recognition.
- **Fine-Tuned Embeddings**: Train `LaBSE` or `indic-sbert` on Bangla-English corpora.
- **Improved Cleaning**: Integrate Bangla spell checkers (`bnltk`, `bangla-stemmer`).
- **Hybrid Retrieval**: Combine Pinecone with BM25 for semantic + keyword matching.
- **Metadata Storage**: Store chunk metadata (page number, confidence) for debugging.

---

## 👩‍💻 Author

- [ ] **Shakil Ahamed**\
       Crafting innovative solutions with code and curiosity! 🚀
