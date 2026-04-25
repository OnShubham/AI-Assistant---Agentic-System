from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(file_path: str) -> list:
    # 1. Load Document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Split into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],  # Split by paragraphs, then lines, then sentences
        chunk_size=500,  # Max characters per chunk
        chunk_overlap=50,  # Overlap to maintain context
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
