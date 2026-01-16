import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# 1. Setup LLM
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Ministral-8B-Instruct-2410"

# Base endpoint
llm_endpoint = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    task="text-generation", # The endpoint itself handles text
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.5,
    max_new_tokens=512,
)

# Chat wrapper (this handles the "conversational" logic)
chat_model = ChatHuggingFace(llm=llm_endpoint)

# 2. Updated Prompt Template (Chat style)
# This format automatically wraps your input in the required Mistral tags
prompt = ChatPromptTemplate.from_template("""
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say you don't know. Do NOT make up answers.

Context:
{context}

Question:
{question}
""")

# 3. Load Vectorstore (Assuming paths are correct)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# 4. Simplified RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

if __name__ == "__main__":
    user_query = input("Write your query: ")
    result = rag_chain.invoke(user_query)
    print("RESULT:\n", result)