import os

# from langchain.document_loaders import UnstructuredPDFLoader,  PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from app.config import PINECONE_API_KEY, OPENAI_API_KEY, DATA_DIR
from uuid import uuid4
import hashlib
import unicodedata
from PIL import Image
import pytesseract
import fitz


def generate_stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


import re


def clean_bangla_text(text):
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Normalize Unicode (important for Bengali modifiers)
    text = unicodedata.normalize("NFKC", text)

    # Remove unwanted non-Bangla characters
    text = re.sub(r"[^\u0980-\u09FF\s.,?!ঃ]", "", text)

    # Fix broken conjuncts (e.g., remove space before "্")
    text = text.replace("্ ", "্")

    return text.strip()


class Chatbot:
    def __init__(
        self,
        pinecone_api_key: str,
        openai_api_key: str,
        index_name: str = "assaignment",
        # embedding_model: str = "intfloat/multilingual-e5-base",
        llm_model: str = "gpt-4",
        pinecone_region: str = "us-east-1",
        pinecone_cloud: str = "aws",
        retriever_k: int = 5,
        temperature: float = 0,
    ):
        # Set environment variable for OpenAI
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name

        # Create index if not exists
        if index_name not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=3072,  # Dimension for OpenAI embeddings
                metric="cosine",
                spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region),
            )

        # Initialize embeddings and vectorstore

        # self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",  # Use OpenAI's text-embedding-3-large model
        )
        self.vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            pinecone_api_key=pinecone_api_key,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": retriever_k})

        # Initialize LLM
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

        # Prompt template

        template = """
You are a helpful and intelligent assistant for a Retrieval-Augmented Generation (RAG) system.

You will receive user questions in either Bangla or English.
Use **only** the retrieved context provided below to answer the question accurately.
If the answer is not explicitly available, try to infer it **reasonably** from the context.
If no answer can be found or inferred, respond with:
- "দুঃখিত, এই প্রশ্নের উত্তর পাওয়া যায়নি।" (for Bangla questions), or
- "Sorry, I couldn't find the answer to your question." (for English questions).

Always respond **in the same language** as the user's question.

Context to search through:
{context}

Chat History:
{chat_history}

Question: {question}

Instructions:
- If the question can be answered by **a single name, phrase, number, or entity**, return the **shortest possible answer** — ideally in **1–3 words**, no extra sentence.
- If the question **requires explanation or description**, then return a **clear full sentence**.
- Never include unnecessary repetition of question terms in the answer.
- Focus on precision and brevity.
- Always respond **in the same language** as the user's question.

Answer:
"""

        self.prompt = PromptTemplate.from_template(template)

    def format_docs(self, docs):
        cleaned_texts = []
        for doc in docs:
            cleaned = clean_bangla_text(doc.page_content)
            cleaned_texts.append(cleaned)

        return "\n\n".join(cleaned_texts)

    def get_answer(self, question: str, chat_history) -> str:
        # Retrieve relevant docs
        docs = self.retriever.invoke(question)
        print("retrival docs---------", docs)
        context = self.format_docs(docs)
        print("similar context-------- ", context)

        # Format full prompt
        full_prompt = self.prompt.format(
            context=context,
            chat_history=chat_history,
            question=question,
        )

        # Get answer from LLM
        response = self.llm.invoke(full_prompt)
        answer = response.content.strip()
        updated_history = chat_history + [{"question": question, "answer": answer}]

        return answer, updated_history

    def chat_loop(self):
        print("Chatbot is ready! Press Ctrl+C to stop.")
        try:
            while True:
                question = input("\nYou: ").strip()
                if not question:
                    print("Please ask something.")
                    continue

                answer = self.get_answer(question)
                print(f"Bot: {answer}")

        except KeyboardInterrupt:
            print("\nExiting chatbot. Bye!")
        except Exception as e:
            print(f"Error: {e}")

    def insert_docs_to_pinecone(self, filepath):
        doc = fitz.open(filepath)
        all_text = ""
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img, lang="ben")
            cleaned = clean_bangla_text(page_text)
            all_text += cleaned + "\n"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=50, separators=["\n", "।", ".", " "]
        )

        documents = text_splitter.create_documents([all_text])
        self.vectorstore.add_documents(documents)
        print(
            f"Inserted {len(documents)} chunks into Pinecone index '{self.index_name}'."
        )


# Usage
if __name__ == "__main__":
    bot = Chatbot(
        pinecone_api_key=PINECONE_API_KEY,
        openai_api_key=OPENAI_API_KEY,
    )
    data_path = os.path.join(DATA_DIR, "HSC26_Bangla1st_Paper.pdf")
    # bot.insert_docs_to_pinecone(data_path)
    bot.chat_loop()
