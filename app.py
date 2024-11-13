import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io
import fitz
from PIL import Image
from langchain.docstore.document import Document
import pytesseract
import pandas as pd
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(pdf_file_path):
    
    all_text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text
        all_text += "\n\n"
    return all_text

def extract_images_from_pdf(pdf_file_path):
    
    all_images = []
    pdf_document = fitz.open(pdf_file_path)
    
    for page_num in range(pdf_document.page_count):
        page_images = []
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            page_images.append(image)
        
        all_images.append(page_images)
    
    pdf_document.close()
    return all_images


def extract_tables_from_pdf(pdf_file_path):
    
    all_tables = []
    with pdfplumber.open(pdf_file_path) as pdf:
        tables = []
        for page in pdf.pages:
            table_settings = {
                "vertical_strategy": "lines",  
                "horizontal_strategy": "lines",
                "snap_tolerance": 4
            }
            for table in page.extract_tables(table_settings):
                tables.append(table)
        all_tables.append(tables)
    return all_tables

def get_text_chunks(text):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_combined_vector_store(text_chunks, table_chunks, image_chunks):
    
    text_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", maxConcurrency=100)
    docs = [Document(page_content=chunk) for chunk in text_chunks + table_chunks + image_chunks]
    vector_store = FAISS.from_documents(docs, text_embeddings)
    vector_store.save_local("multi_vector_faiss_index")

def retrieve_from_combined_store(query, vector_store):
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", maxConcurrency=100)
    db = FAISS.load_local("multi_vector_faiss_index", embeddings, allow_dangerous_deserialization=True)
    relevant_chunks = db.similarity_search(query)
    return relevant_chunks

def get_conversational_chain():
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def summarize_content(text):
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    input_data = [{"role": "user", "content": f"Summarize the following text:\n\n{text}"}]
    response = model.generate(input_data, history=None)
    return response["content"]


def summarize_tables(tables):

    summaries = []
    for table in tables:
        df = pd.DataFrame(table)
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list) or isinstance(x, dict)).any():
                df[col] = df[col].apply(str)
  
        summary = df.describe(include='all').to_string()

        categorical_cols = df.select_dtypes(include=['object'])
        if not categorical_cols.empty:
            for col in categorical_cols:
                n_unique = df[col].nunique()
                most_frequent = df[col].mode().iloc[0]
                summary += f"\n  - Column '{col}': {n_unique} unique values, most frequent: {most_frequent}"

        summaries.append(summary)
    return summaries

def summarize_images_with_ocr(images):
    
    all_text = []
    for page_images in images:
        for image in page_images:
            text = pytesseract.image_to_string(image)
            all_text.append(text)
    return ("\n".join(all_text))

def user_input(user_question):
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", maxConcurrency=100)
    new_db = FAISS.load_local("multi_vector_faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:")
    st.write(response["output_text"])

def main():
    
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.title("RAG Pipeline for Annual Report Analysis")
    
    pdf_file_path = "data/SriLankan_Airlines_Annual_Report.pdf"
    
    if not os.path.exists(pdf_file_path):
        st.error("File not found. Please check the path and try again.")
        return
    else:
        user_question = st.text_input("Ask a Question from the PDF")
        if not user_question:
            st.write("Note: Press \"Enter\" to ask the question'")
        if user_question:
            user_input(user_question)
            
    if st.button("Process PDF"):
        with st.spinner("Processing the PDF..."):
            raw_text = extract_text_from_pdf(pdf_file_path)
            tables = extract_tables_from_pdf(pdf_file_path)
            images = extract_images_from_pdf(pdf_file_path)

        with st.spinner("Summarizing content..."):
            table_summary = summarize_tables(tables)
            image_summary = summarize_images_with_ocr(images)
            table_summary_text = "\n".join(table_summary)
            
        with st.spinner("Creating vector store..."):
            text_chunks = get_text_chunks(raw_text)
            table_chunks = get_text_chunks(table_summary_text)
            image_chunks = get_text_chunks(image_summary)
            
            get_combined_vector_store(text_chunks , table_chunks , image_chunks)
        
        st.success("Processing Done")


if __name__ == "__main__":
    main()
