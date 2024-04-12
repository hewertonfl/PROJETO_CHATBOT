from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import os


class CreateEmbeddings:
    def __init__(self):
        """Initialize the class with the persist directory and the embedding model."""
        self.persist_dir = "db/rag_net_eng"
        self.persist_dir = "db/rag_acionamentos_eng"
        self.embedding = OllamaEmbeddings(model='nomic-embed-text')
        self.docs_dir = os.path.join(os.getcwd(), "documents")
        self.chunk_size = 2500
        self.chunk_overlap = 0.3*self.chunk_size
        self.urls = [
            "https://ollama.com/",
            "https://ollama.com/blog/windows-preview",
            "https://ollama.com"
        ]

    def get_website_content(self):
        """Load content from a website and split it into chunks."""
        urls = self.urls
        documents = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in documents for item in sublist]
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(docs_list)
        return docs

    def get_pdf_content(self):
        """Load PDFs from a directory and split them into chunks."""
        print("Loading PDFs...")
        file_loader = PyPDFDirectoryLoader(self.docs_dir)
        docs = file_loader.load()
        docs = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap).split_documents(docs)
        return docs

    def get_embedding_vector(self):
        print("Embedding vectors...")
        docs = self.get_pdf_content()
        vectordb = None
        vectordb = Chroma.from_documents(
            documents=docs, embedding=self.embedding, persist_directory=self.persist_dir)
        vectordb.persist()
        print("Done!")


# CreateEmbeddings().get_pdf_content()
CreateEmbeddings().get_embedding_vector()
