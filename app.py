from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------
# Page Config
# --------------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------
# Custom CSS
# --------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

.title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#4CAF50;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Header
# --------------------------------
st.markdown(
    '<p class="title">🤖 RAG Chatbot</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">LangChain + ChromaDB + Mistral AI</p>',
    unsafe_allow_html=True
)

# --------------------------------
# Load Models
# --------------------------------
@st.cache_resource
def load_components():

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2603"
    )

    return retriever, llm

retriever, llm = load_components()

# --------------------------------
# Prompt
# --------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful assistant.
        Answer only from the provided context.
        If the answer is not found,
        say "I don't know".
        """
    ),
    (
        "user",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])

# --------------------------------
# Session State
# --------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------
# User Input
# --------------------------------
if query := st.chat_input("Ask something..."):

    st.session_state.messages.append(
        {"role":"user","content":query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    response = llm.invoke(
        prompt.format_prompt(
            context=context,
            question=query
        ).to_messages()
    )

    with st.chat_message("assistant"):
        st.markdown(response.content)

        with st.expander("Retrieved Context"):
            for i, doc in enumerate(docs):
                st.write(f"### Document {i+1}")
                st.write(doc.page_content)
                st.write(doc.metadata)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response.content
        }
    )