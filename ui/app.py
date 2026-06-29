import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Procure-AI",
    page_icon="🖥️",
    layout="centered"
)

st.title("🖥️ Procure-AI")
st.caption("IT Product Advisor powered by Agentic RAG")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📦 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['name']}** — {s['price_thb']:,.0f} THB `{s['category']}`")

# Chat input
if query := st.chat_input("Ask about IT products..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"query": query}
                )
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]

                st.markdown(answer)
                with st.expander("📦 Sources"):
                    for s in sources:
                        st.markdown(f"**{s['name']}** — {s['price_thb']:,.0f} THB `{s['category']}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error: {e}")