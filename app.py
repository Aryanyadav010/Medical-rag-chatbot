import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")
st.title("🩺 Medical RAG Chatbot")
st.caption("Answers questions using a medical book as its knowledge source.")

# Load keys from Streamlit secrets (never hardcode them)
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot", embedding=embeddings
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)
    return retriever, llm

retriever, llm = load_components()

PROMPT = """You are a medical information assistant. Use ONLY the context below
to answer the question. If the answer is not in the context, say you don't know.
Keep the answer clear and concise (max 5 sentences).

Context:
{context}

Question: {question}

Answer:"""

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask a medical question, e.g. What is acne?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the book..."):
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            answer = llm.invoke(
                PROMPT.format(context=context, question=question)
            ).content
        st.markdown(answer)
        with st.expander("Sources"):
            for d in docs:
                page = d.metadata.get("page", "?")
                st.write(f"Page {page}: {d.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.warning(
    "⚠️ For educational purposes only. Not a substitute for professional medical advice."
)
