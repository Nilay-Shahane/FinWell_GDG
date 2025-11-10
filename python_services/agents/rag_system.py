"""
RAG (Retrieval-Augmented Generation) System
Supports CSV, PDF, and TXT document ingestion with FAISS vector storage.
"""

import os
from typing import List
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .llm_main import llm


class BasicRAG:
    """A basic RAG system using Google embeddings and FAISS vector store."""
    
    # File loader mapping
    LOADERS = {
        '.csv': CSVLoader,
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
    }
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the RAG system.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
        """
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.qa_chain = None

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document based on file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of loaded documents
            
        Raises:
            ValueError: If file type is unsupported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        loader_class = self.LOADERS.get(ext)
        if not loader_class:
            supported = ', '.join(self.LOADERS.keys())
            raise ValueError(f"Unsupported file type: {ext}. Supported types: {supported}")
        
        loader = loader_class(file_path)
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} document(s) from {file_path}")
        return documents

    def split_text(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        print(f"✓ Split into {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document]) -> None:
        """
        Create FAISS vector store from document chunks.
        
        Args:
            chunks: List of document chunks to vectorize
        """
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("✓ Vector store created")

    def create_qa_chain(self, top_k: int = 3) -> None:
        """
        Create the question-answering chain using LCEL.
        
        Args:
            top_k: Number of document chunks to retrieve for each query
            
        Raises:
            ValueError: If vectorstore hasn't been created
        """
        if not self.vectorstore:
            raise ValueError("Vector store not found. Run create_vectorstore() first.")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
        )
        
        self.qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ QA chain ready")

    def ask(self, query: str, verbose: bool = True) -> str:
        """
        Ask a question and get an answer from the RAG system.
        
        Args:
            query: Question to ask
            verbose: Whether to print question and answer
            
        Returns:
            Answer string
            
        Raises:
            ValueError: If QA chain hasn't been created
        """
        if not self.qa_chain:
            raise ValueError("QA chain not created. Run create_qa_chain() first.")
        
        answer = self.qa_chain.invoke(query)
        
        if verbose:
            print(f"\nQuestion: {query}")
            print(f"Answer: {answer}")
        
        return answer

    def setup_from_file(self, file_path: str, top_k: int = 3) -> None:
        """
        Convenience method to set up the entire RAG pipeline from a file.
        
        Args:
            file_path: Path to the document file
            top_k: Number of chunks to retrieve per query
        """
        documents = self.load_document(file_path)
        chunks = self.split_text(documents)
        self.create_vectorstore(chunks)
        self.create_qa_chain(top_k=top_k)
        print("\n✓ RAG system ready for queries!")