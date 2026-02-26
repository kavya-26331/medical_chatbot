# frontend/streamlit_app.py
# frontend/streamlit_app.py
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="AI Medical Chatbot",
    layout="wide",
    page_icon="🩺"
)

# Custom CSS for cyber medical dark theme
st.markdown("""
<style>
/* General background and font */
body, .stApp {
    background-color: #0f111a;
    color: #e0e0e0;
    font-family: 'Arial', sans-serif;
}

/* Headers */
h1, h2, h3, h4, h5 {
    color: #00f0ff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #121526;
    color: #e0e0e0;
    padding: 20px;
}

/* Buttons */
.stButton>button {
    background-color: #00f0ff;
    color: #0f111a;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    width: 100%;
    padding: 10px;
}

.stButton>button:hover {
    background-color: #00c8e0;
    color: #000;
}

/* Text areas */
.stTextArea>div>div>textarea {
    background-color: #1a1c2b;
    color: #e0e0e0;
    border-radius: 10px;
    padding: 12px;
    font-size: 16px;
}

/* Chat bubbles */
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
}

.user-msg {
    background-color: #001f33;
    border-left: 4px solid #00f0ff;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 8px;
}

.bot-msg {
    background-color: #1a1c2b;
    border-left: 4px solid #00ff9f;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 8px;
}

.source-card {
    background-color: #121526;
    border-left: 4px solid #ffaa00;
    padding: 8px;
    margin-bottom: 10px;
    border-radius: 8px;
}

/* Scrollbar styling */
.chat-container::-webkit-scrollbar {
    width: 8px;
}
.chat-container::-webkit-scrollbar-thumb {
    background-color: #00f0ff;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'>🩺 AI Medical Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #00f0ff'>", unsafe_allow_html=True)

# Sidebar - collapsible sections
with st.sidebar:
    st.subheader("📄 Upload Medical Documents")
    with st.expander("Upload Instructions"):
        st.markdown("""
        - Only .txt files supported.  
        - Multiple files allowed.  
        - After upload, click **Ingest** to process documents.
        """)
    uploaded = st.file_uploader("Select text files", type=["txt"], accept_multiple_files=True)
    
    if st.button("Ingest Files"):
        if not uploaded:
            st.warning("Select one or more files first.")
        else:
            clear_resp = requests.post(f"{API_URL}/clear")
            if clear_resp.ok:
                st.success("✅ Vector DB cleared!")
            else:
                st.error(f"Failed to clear DB: {clear_resp.text}")
            
            for f in uploaded:
                files = {"file": (f.name, f.getvalue(), "text/plain")}
                resp = requests.post(f"{API_URL}/upload_doc", files=files, data={"source_name": f.name})
                if resp.ok:
                    st.success(f"Ingested {f.name}: {resp.json().get('message')}")
                else:
                    st.error(f"Failed to ingest {f.name}: {resp.text}")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input (floating style)
st.markdown("<h2>💬 Chat with AI</h2>", unsafe_allow_html=True)

# Chat container
chat_html = "<div class='chat-container'>"
for msg in reversed(st.session_state.messages):
    chat_html += f"<div class='user-msg'><strong>User:</strong> {msg['user']}</div>"
    chat_html += f"<div class='bot-msg'><strong>Assistant:</strong> {msg['bot']}</div>"
    if msg.get("sources"):
        for s in msg["sources"]:
            chat_html += f"<div class='source-card'>📌 {s}</div>"
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# Input area at bottom
query = st.text_area("Type your question here...", height=100)

if st.button("🚀 Ask AI"):
    if query.strip() == "":
        st.warning("Type a question first!")
    else:
        payload = {"query": query}
        with st.spinner("🤖 Thinking..."):
            resp = requests.post(f"{API_URL}/chat", json=payload, timeout=300)
        if resp.ok:
            res = resp.json()
            st.session_state.messages.append({
                "user": query,
                "bot": res["answer"],
                "sources": res.get("sources", [])
            })
            st.rerun()  # Refresh chat to show latest
        else:
            st.error(f"Error: {resp.text}")







