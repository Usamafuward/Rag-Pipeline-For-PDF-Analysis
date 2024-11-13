######
This project is a RAG (Retrieval-Augmented Generation) pipeline that processes PDF documents, extracts structured and unstructured content, and enables users to query the extracted information through a conversational interface. It integrates tools for text, table, and image extraction, embeddings for multi-modal data, and a question-answering system powered by Google Generative AI.

######
Installation -----------

1. Clone the Repository

    git clone https://github.com/Usamafuward/Rag-Pipeline-For-PDF-Analysis.git
    cd Rag-Pipeline-For-PDF-Analysis

3. Create a Virtual Environment

    python -m venv venv
    ./venv/Scripts/activate

4. Install Dependencies

    pip install -r requirements.txt

5. Set Up Environment Variables

    Create a .env file in the project root and add GOOGLE_API_KEY=<your-google-generative-ai-api-key>

######
Run The Application -----------

    streamlit run main.py
