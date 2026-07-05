import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Create documents
docs = [
    Document(
        page_content="LangChain is a framework used for building applications with large language models.",
        metadata={"source": "LangChain"}
    ),

    Document(
        page_content="LangChain is a framework for LLM applications.",
        metadata={"source": "LangChain"}
    ),
    Document(
        page_content="Chroma is a vector database for LLM applications.",
        metadata={"source": "Chroma"}
    ),
    Document(
        page_content="HuggingFace is a platform for machine learning models.",
        metadata={"source": "HuggingFace"}
    )
]

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_PATH = "chroma-db"

# Create the vector DB only if it doesn't exist
if not os.path.exists(DB_PATH):
    print("Creating vector database...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("Database created successfully!")

else:
    print("Loading existing vector database...")

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

# Perform similarity search
results = vectorstore.similarity_search(
    "What is LangChain?",
    k=2 # k means the number of similar documents to retrieve
)

print("\nSearch Results:\n")

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print()

retriever = vectorstore.as_retriever()
docs = retriever.invoke("What is LangChain?")