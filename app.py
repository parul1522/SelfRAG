import streamlit as st
import tempfile

from grade import process_document, grap

st.set_page_config(
    page_title="RAG Assistant",
    layout="wide"
)

st.title("Self-Correcting RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:

    st.subheader("Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "docx"]
    )

    pasted_text = st.text_area(
        "Paste text",
        height=200
    )

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f"_{uploaded_file.name}"
        ) as tmp:

            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        with st.spinner("Processing document..."):
            st.session_state.vectorstore = process_document(
                temp_path
            )

        st.success("Document loaded successfully")

    elif pasted_text:

        with st.spinner("Processing text..."):
            st.session_state.vectorstore = process_document(
                pasted_text
            )

        st.success("Text loaded successfully")

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input(
    "Ask a question about your document"
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vectorstore is None:

        response = (
            "Please upload a document or paste text first."
        )

        with st.chat_message("assistant"):
            st.markdown(response)

    else:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = grap.invoke(
                    {
                        "question": query,
                        "original_question": query,
                        "docs": [],
                        "answer": "",
                        "retry_count": 0,
                        "vectorstore": st.session_state.vectorstore
                    }
                )

                response = result["answer"]

            st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )