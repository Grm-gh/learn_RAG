from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = chroma.Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
llm = ChatMistralAI(model="mistral-small-2603")

prompt_template = ChatPromptTemplate.from_messages([
    ("system","""You are a helpful assistant that answers questions based on the context provided.
If the context does not contain the answer, respond with "I don't know."""),
    ("user", "Use the following context to answer the question.\n\n{context}\n\nQuestion: {question}")
])

print("Welcome to the LangChain Mistral Chatbot! Type 'exit' to quit.\n")
print("press 0 to exit")
while True:
    user_input = input("You: ")

    if user_input == "exit" or user_input == "0":
        break

    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])

    response = llm.invoke(
        prompt_template.format_prompt(context=context, question=user_input).to_messages()
    )

    print("Bot:", response.content)
