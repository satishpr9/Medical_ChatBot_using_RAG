import os 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH = 'data/'

def load_pdf_files(data):
    loader=DirectoryLoader(data,
            glob='*.pdf',
            loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents=load_pdf_files(DATA_PATH)
print(f"Loaded {len(documents)} documents")

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(documents)
print(f"Created {len(text_chunks)} text chunks")


def get_embeddings(text_chunks):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
embeddings=get_embeddings(text_chunks)

DB_FAISS_PATH='vectorstore/db_faiss'
# if not os.path.exists(DB_FAISS_PATH):
#     vectorstore=FAISS.from_documents(text_chunks, embeddings)
#     vectorstore.save_local(DB_FAISS_PATH)
# else:
#     vectorstore=FAISS.load_local(DB_FAISS_PATH, embeddings)
# print(f"Loaded vectorstore from {DB_FAISS_PATH}")

DB_FAISS_PATH='vectorstore/db_faiss'
db=FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS_PATH)
