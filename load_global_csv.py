import os
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CSV_FILE_PATH = "active_projects.csv"

def load_csv_file():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error : {CSV_FILE_PATH} not found")
        return
    
    print(f"LOading {CSV_FILE_PATH}...")
    loader = CSVLoader(
        file_path=CSV_FILE_PATH, 
        encoding="utf-8-sig",
        metadata_columns=["Project URL","Categories","Technology","Priority", "Description"],
        content_columns=["Description", "Technology", "Categories", "Project URL"] 
    )
    docs = loader.load()
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="global_csv_data"
    )
    print(f"Successfully loaded {len(docs)} rows into chroma db")
    
if __name__ == "__main__":
    load_csv_file()