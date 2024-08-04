import pandas as pd
import streamlit as st
import pymupdf  # Ensure this is installed
import tempfile
from langchain.chains import RetrievalQA
import io
from PIL import Image
import requests
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Proposal Toolkit",
    page_icon="https://raw.githubusercontent.com/scooter7/ask-multiple-pdfs/main/ACE_92x93.png"
)

# Hide the toolbar
hide_toolbar_css = """
<style>
    .css-14xtw13.e8zbici0 { display: none !important; }
</style>
"""
st.markdown(hide_toolbar_css, unsafe_allow_html=True)

# Main content
header_html = """
<div style="text-align: center;">
    <h1 style="font-weight: bold;">Proposal Toolkit</h1>
    <img src="https://www.carnegiehighered.com/wp-content/uploads/2021/11/Twitter-Image-2-2021.png" alt="Icon" style="height:200px; width:500px;">
    <p align="left">Find and develop proposal resources. The text entry field will appear momentarily.</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Import fitz from pymupdf
import fitz

# Function to load PDF from file-like object
@st.cache_data
def load_pdf(file):
    return fitz.open(stream=file, filetype="pdf")

# Function to download PDF files from the GitHub repository
def download_pdf_files():
    repo_url = "https://api.github.com/repos/scooter7/demo_highlight_pdf_streamlit/contents/Docs"
    response = requests.get(repo_url)
    pdf_files = []
    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            if item['name'].endswith('.pdf'):
                pdf_files.append(item['download_url'])
    return pdf_files

# Custom function to extract document objects from file URL
def extract_documents_from_url(file_url):
    response = requests.get(file_url)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(response.content)
    temp_file.close()
    
    documents = load_pdf(temp_file.name)
    return documents

def find_pages_with_excerpts(doc, excerpts):
    pages_with_excerpts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for excerpt in excerpts:
            text_instances = page.search_for(excerpt)
            if text_instances:
                pages_with_excerpts.append(page_num)
                break
    return pages_with_excerpts if pages_with_excerpts else [0]

@st.cache_resource
def get_llm():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0, 
        openai_api_key=st.secrets["openai"]["api_key"]
    )
    return llm

@st.cache_resource
def get_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", 
        openai_api_key=st.secrets["openai"]["api_key"]
    )
    return embeddings

@st.cache_resource
def get_qa(_documents):
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    texts = text_splitter.split_documents(_documents)
    
    db = Qdrant.from_documents(
        documents=texts,
        embedding=get_embeddings(),
        collection_name="my_documents",
        location=":memory:"
    )
    
    retriever = db.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 2, "lambda_mult": 0.8}
    )
    
    qa = RetrievalQA.from_chain_type(
        get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    )
    return qa

def get_highlight_info(doc, excerpts):
    annotations = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for excerpt in excerpts:
            text_instances = page.search_for(excerpt)
            if text_instances:
                for inst in text_instances:
                    annotations.append({
                        "page": page_num + 1,
                        "x": inst.x0,
                        "y": inst.y0,
                        "width": inst.x1 - inst.x0,
                        "height": inst.y1 - inst.y0,
                        "color": "red",
                    })
    return annotations

custom_template = """
    Use the following pieces of context to answer the user question. If you
    don't know the answer, just say that you don't know, don't try to make up an
    answer.

    {context}

    Question: {question}

    Please provide your answer in the following JSON format: 
    {{
        "answer": "Your detailed answer here",
        "sources": "Direct sentences or paragraphs from the context that support 
            your answers. ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. DO NOT 
            ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING."
    }}
    
    The JSON must be a valid json format and can be read with json.loads() in
    Python. Answer:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=custom_template, input_variables=["context", "question"]
)

# Main functionality
def main():
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_pdf_text' not in st.session_state:
        st.session_state.uploaded_pdf_text = None
    if 'institution_name' not in st.session_state:
        st.session_state.institution_name = None
    if 'pdf_keywords' not in st.session_state:
        st.session_state.pdf_keywords = []

    pdf_files = download_pdf_files()
    
    if pdf_files:
        file_url = pdf_files[0]  # Automatically use the first PDF

        with st.spinner("Processing file..."):
            documents = extract_documents_from_url(file_url)
            response = requests.get(file_url)
            st.session_state.doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")

        if documents:
            qa = get_qa(documents)

            # Display chat messages
            st.subheader("Chat History")
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**You**: {msg['content']}")
                else:
                    st.markdown(f"**Assistant**: {msg['content']}")

            # Chat input
            user_input = st.text_input("Enter your query to search in documents and craft new content", key="user_input")
            if st.button("Send"):
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                st.markdown(f"**You**: {user_input}")

                with st.spinner("Generating response..."):
                    try:
                        result = qa.invoke({"query": user_input})
                        parsed_result = json.loads(result['result'])

                        answer = parsed_result['answer']
                        sources = parsed_result['sources']
                        sources = sources.split(". ") if pd.notna(sources) else []

                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer}
                        )
                        st.markdown(f"**Assistant**: {answer}")

                        st.session_state.sources = sources
                        st.session_state.chat_occurred = True

                    except json.JSONDecodeError:
                        st.error(
                            "There was an error parsing the response. Please try again."
                        )

                if st.session_state.get("chat_occurred", False):
                    doc = st.session_state.doc
                    st.session_state.total_pages = len(doc)
                    if "current_page" not in st.session_state:
                        st.session_state.current_page = 0

                    pages_with_excerpts = find_pages_with_excerpts(doc, st.session_state.sources)

                    if "current_page" not in st.session_state:
                        st.session_state.current_page = pages_with_excerpts[0]

                    st.session_state.cleaned_sources = st.session_state.sources
                    st.session_state.pages_with_excerpts = pages_with_excerpts

                    st.markdown("### PDF Preview with Highlighted Excerpts")

                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        if st.button("Previous Page") and st.session_state.current_page > 0:
                            st.session_state.current_page -= 1
                    with col2:
                        st.write(
                            f"Page {st.session_state.current_page + 1} of {st.session_state.total_pages}"
                        )
                    with col3:
                        if (
                            st.button("Next Page")
                            and st.session_state.current_page < st.session_state.total_pages - 1
                        ):
                            st.session_state.current_page += 1

                    page_number = st.session_state.current_page
                    page = doc.load_page(page_number)
                    page_image = page.get_pixmap()

                    st.image(page_image.tobytes(), use_column_width=True)

                    highlights = get_highlight_info(doc, st.session_state.sources)
                    if highlights:
                        st.markdown("**Highlighted Excerpts:**")
                        for highlight in highlights:
                            st.markdown(f"**Page {highlight['page']}** - Highlighted at ({highlight['x']}, {highlight['y']})")

                    st.write("**End of Document Highlights**")
        else:
            st.error("No documents found.")
    else:
        st.error("No PDFs available for download.")
    
if __name__ == "__main__":
    main()
