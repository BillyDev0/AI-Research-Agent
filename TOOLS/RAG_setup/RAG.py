from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

embedding = OllamaEmbeddings(model="nomic-embed-text:latest")

file="D:/AI_Research_Agent/TOOLS/RAG_setup/vector_db"
def embed_file():
    if os.path.exists(file):
        return Chroma(
            embedding_function=embedding,
            persist_directory=file
        )

    with open("D:/AI_Research_Agent/TOOLS/RAG_setup/documents.txt") as f:
        text=f.read()

    chunk=text.split("\n\n\n")

    return Chroma.from_texts(
        embedding=embedding,
        texts=chunk,
        persist_directory=file
    )

def similarity_search(prompt):
    vector_db=embed_file()
    retriever=vector_db.as_retriever()
    context=retriever.invoke(prompt)
    return context[0]

# print(similarity_search("apa saja fitur yang dimiliki oleh toko pro"))