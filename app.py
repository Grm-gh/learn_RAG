import time
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# =====================================================
# PAGE CONFIG - Set wide layout and clean title
# =====================================================
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# ENHANCED UI STYLING (CSS)
# =====================================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }
    
    /* Metrics Styling */
    .metric-container {
        background: #1e293b;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
    }

    /* Context Expander */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 8px;
    }
    
    /* Typography */
    h1 { color: #f8fafc !important; }
    .stMarkdown, .stText { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER SECTION
# =====================================================
st.title("🤖 RAG AI Assistant")
st.markdown("##### *LangChain • ChromaDB • Mistral AI*")
st.markdown("---")

# =====================================================
# SIDEBAR - Structured Configuration Panel
# =====================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Using columns inside sidebar for a compact look
    st.info("**Active Stack**")
    st.write("• **LLM:** Mistral Small")
    st.write("• **Vector DB:** Chroma")
    st.write("• **Embedding:** MiniLM")
    st.write("• **Search:** MMR")
    
    st.divider()
    
    st.caption("Pipeline Architecture")
    st.graphviz_chart('''
        digraph {
            node [shape=box, style=rounded, fontname="sans-serif", fontsize=10];
            User -> Retriever -> ChromaDB -> Context -> Mistral -> Answer;
        }
    ''')

# =====================================================
# LOAD COMPONENTS (Cached)
# =====================================================
@st.cache_resource
def load_models():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma-db", embedding_function=embedding)
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    llm = ChatMistralAI(model="mistral-small-2603")
    return retriever, llm

retriever, llm = load_models()

# =====================================================
# CHAT INTERFACE LOGIC
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Retrieving and generating..."):
        start = time.time()
        docs = retriever.invoke(question)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based on context: {context}"),
            ("user", "{question}")
        ])
        
        response = llm.invoke(prompt.format_prompt(context=context_text, question=question).to_messages())
        end = time.time()

    # Display Response
    with st.chat_message("assistant"):
        st.markdown(response.content)
        
        # Display Metrics cleanly using columns
        cols = st.columns(3)
        metrics = [
            ("Documents", len(docs)),
            ("Latency", f"{end-start:.2f}s"),
            ("Tokens", len(response.content.split()))
        ]
        
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">{metrics[i][0]}</div>
                    <div class="metric-value">{metrics[i][1]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Context Expander
        with st.expander("📚 View Retrieved Context"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}**")
                st.info(doc.page_content)
    
    st.session_state.messages.append({"role": "assistant", "content": response.content})