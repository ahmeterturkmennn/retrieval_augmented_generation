from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import json
import chromadb
from chromadb.utils import embedding_functions
def preprocess(chunk_size,chunk_overlap,embedding_model,chunks_file,pdf_path):
    splitter = RecursiveCharacterTextSplitter(        
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]  
    )
    
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(embedding_model)
    
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except AttributeError:
        pass
    
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=chromadb.Settings(anonymized_telemetry=False)
    )
    collection_name = f"cv_book_cs{chunk_size}_ov{chunk_overlap}"
    collection = client.get_or_create_collection(
       name=collection_name,
       embedding_function=embedding_fn
   )
    if Path(chunks_file).exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        
        reader = PdfReader(pdf_path)
        
        text = ""
        
        for page in reader.pages:
            page_text = page.extract_text()
        
            if page_text:
                text += page_text + "\n"
        
        chunks = splitter.split_text(text)
        
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    if collection.count() != len(chunks):
       if collection.count() > 0:
           client.delete_collection(collection_name)
           collection = client.get_or_create_collection(
               name=collection_name,
               embedding_function=embedding_fn
           )
       collection.add(
           documents=chunks,
           ids=[f"chunk_{i}" for i in range(len(chunks))]
       )
    
    return chunks,collection
