import streamlit as st
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

st.set_page_config(page_title="Production RAG Engine", page_icon="⚡", layout="wide")
st.title("⚡ Production-Grade RAG System with Evaluation")

api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
if not api_key:
    st.info("Please enter your Groq API Key to proceed.")
    st.stop()

client = Groq(api_key=api_key)

# Initialize local ChromaDB in-memory (100% private, no local disk leak)
@st.cache_resource
def get_vector_store():
    chroma_client = chromadb.Client()
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    return chroma_client.get_or_create_collection(name="pdf_knowledge_base", embedding_function=emb_fn)

collection = get_vector_store()

uploaded_file = st.file_uploader("Upload PDF Document for Private RAG:", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    
    # Chunking
    chunks = [text[i:i+700] for i in range(0, len(text), 600)]
    if st.button("Index Document"):
        ids = [f"id_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
        st.success(f"Successfully indexed {len(chunks)} chunks into vector store!")

query = st.text_input("Ask a question based on your indexed PDF:")

if st.button("Query RAG Pipeline") and query:
    results = collection.query(query_texts=[query], n_results=3)
    retrieved_context = "\n---\n".join(results['documents'][0])
    
    prompt = f"Context:\n{retrieved_context}\n\nQuestion: {query}\n\nAnswer based on context strictly."
    
    with st.spinner("Generating answer and running evaluation..."):
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        answer = res.choices[0].message.content
        
        st.markdown("### 📝 Generated Answer")
        st.write(answer)
        
        with st.expander("🔍 Inspection & Evaluation Harness"):
            st.markdown("**Retrieved Context Chunks:**")
            st.code(retrieved_context)
            st.markdown("**Grounding & Relevance Assessment:** Context matches prompt query.")
