"""
Embedding documents localy using opensource models from Huggingface
"""

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

# Loading Huggingface api token from .env file ,
# alternatively, you  may load api token using os.environ["HUGGINGFACEHUB_API_TOKEN"]="api token here"
load_dotenv()
os.getenv("HUGGINGFACEHUB_API_TOKEN")

def embed_document(path: str, chunk_size:int, chunk_overlap:int, model_name= "all-MiniLM-L6-v2", separator= "\n\n"):
    loader = TextLoader(
        path, encoding='utf8'
    )
    document = loader.load()

    # Split text into chunks to fit model's context window
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
    texts = text_splitter.split_documents(document)

    # Embed the document using opensource models from huggingface
    # Model: all-MiniLM-L6-v2

    embeddings = HuggingFaceEmbeddings(model_name= model_name)

    #Facebook AI Similarity Search (Faiss)
    vector_store = FAISS.from_documents(texts, embeddings)

    # Saving these embeddings locally ( Optionally you may save them on Vector DBs like Pinecone etc)
    vector_store.save_local("faiss_based_index")
    return embeddings

if __name__ == '__main__':
    embeddings = embed_document(path="./transformer_overview.txt", chunk_size=300, chunk_overlap=10, model_name="all-MiniLM-L6-v2", separator="\n"
                                )
    # read back from local vector store
    new_vector_store = FAISS.load_local("faiss_based_index", embeddings)
    query_string = "what is attention"
    for content in new_vector_store.similarity_search(query_string):
        print(content.page_content)

