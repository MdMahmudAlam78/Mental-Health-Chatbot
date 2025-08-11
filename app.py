# app.py

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(page_title="MENTAL HEALTH CHATBOT", layout="centered")
st.title("🧠 MENTAL HEALTH CHATBOT")
st.markdown("#### 💬 _Ask me anything related to mental health. I’ll answer based on WHO’s mhGAP guidelines._")

# Step 1: Load FAISS vector store
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Step 2: Load Hugging Face model (flan-t5-base)
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Step 3: Prompt template to guide the LLM
prompt_template = """
You are a helpful mental health assistant.
Based on the WHO mhGAP guide context, answer the user's question clearly and concisely in 4-5 lines.
Avoid repeating the same content. If the answer is unclear, recommend consulting a professional.

Context:
{context}

Question:
{question}

Answer:
"""
custom_prompt = PromptTemplate.from_template(prompt_template)

# Step 4: Set up memory and conversation chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# Step 5: Input form with submit button
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_input("Your question:", key="user_input")
    submitted = st.form_submit_button("Submit")

if submitted and user_query:
    with st.spinner("🤖 Thinking..."):
        try:
            result = qa_chain({"question": user_query})
            st.session_state.chat_history.append((user_query, result["answer"]))
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Step 6: Display chat history
if st.session_state.chat_history:
    st.subheader("🗨️ Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")
