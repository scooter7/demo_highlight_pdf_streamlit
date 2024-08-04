import pandas as pd
import streamlit as st
import os
import fitz  # PyMuPDF
import tempfile
from langchain.chains import RetrievalQA
import io
from streamlit_pdf_viewer import pdf_viewer
import json
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

# Set page config
st.set_page_config(page_title="📚 ChatPDF")

# Main content
st.markdown("<h1 style='text-align: center;'>📚 ChatPDF</h1>", unsafe_allow_html=True)
st.subheader("Query documents stored in the repository to get started.")

# Function to load PDF from file-like object
@st.cache_data
def load_pdf(file):
    return fitz.open(stream=file, filetype="pdf")

# Function to download PDF files from the GitHub repository
def download_pdf_files():
    repo_url = "https://github.com/scooter7/demo_highlight_pdf_streamlit/tree/main/Docs"
    response = requests.get(repo_url)
    pdf_files = []
    if response.status_code == 200:
        # Extract links to the PDF files from the HTML content
        links = response.content.decode('utf-8').split('<a href="')
        for link in links:
            if link.endswith('.pdf">'):
                pdf_url = "https://github.com" + link.split('"')[0]
                pdf_files.append(pdf_url)
    return pdf_files

# Custom function to extract document objects from file URL
def extract_documents_from_url(file_url):
    response = requests.get(file_url)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(response.content)
    temp_file.close()
    
    loader = PyPDFLoader(temp_file.name)
    documents = loader.load()
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
pdf_files = download_pdf_files()
if pdf_files:
    selected_pdf = st.selectbox("Select a PDF file to query:", options=pdf_files, key='selected_pdf')

    if selected_pdf:
        file_url = selected_pdf

        with st.spinner("Processing file..."):
            documents = extract_documents_from_url(file_url)
            response = requests.get(file_url)
            st.session_state.doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")

        if documents:
            qa = get_qa(documents)
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                    {"role": "assistant", "content": "Hello! How can I assist you today?"}
                ]

            for msg in st.session_state.chat_history:
                st.chat_message(msg["role"]).write(msg["content"])

            if user_input := st.chat_input("Your message"):
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                st.chat_message("user").write(user_input)

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
                        st.chat_message("assistant").write(answer)

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

                    pages_with_excerpts = find_pages_with_excerpts(doc, sources)

                    if "current_page" not in st.session_state:
                        st.session_state.current_page = pages_with_excerpts[0]

                    st.session_state.cleaned_sources = sources
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
                            and st.session_state.current_page
                            < st.session_state.total_pages - 1
                        ):
                            st.session_state.current_page += 1

                    annotations = get_highlight_info(doc, st.session_state.sources)

                    if annotations:
                        first_page_with_excerpts = min(ann["page"] for ann in annotations)
                    else:
                        first_page_with_excerpts = st.session_state.current_page + 1

                    pdf_viewer(
                        response.content,
                        width=700,
                        height=800,
                        annotations=annotations,
                        pages_to_render=[first_page_with_excerpts],
                    )
