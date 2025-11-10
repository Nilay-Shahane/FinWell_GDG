import os

from .rag_system import BasicRAG
from dotenv import load_dotenv

def rag_pipeline(file_path, query):
    """Complete RAG pipeline: load, chunk, vectorize, and query."""
    # This is fine, but note: this re-initializes the model on every call.
    # For a real app, you might initialize 'rag' once outside.
    rag = BasicRAG() 
    docs = rag.load_document(file_path)
    chunks = rag.split_text(docs)
    rag.create_vectorstore(chunks)
    rag.create_qa_chain() 
    return rag.ask(query)

def get_rag_answer(file_path: str, query: str):
    """Run the RAG pipeline for a given query."""
    
    # --- FIXED ---
    # The file_path and query were hardcoded here, ignoring the
    # arguments. This now correctly uses the arguments passed to it.
    print(f"Starting RAG pipeline for file: {file_path}")
    response = rag_pipeline(file_path, query)
    return response


if __name__ == "__main__":
   
    load_dotenv() 

    # --- UPDATED ---
    # This main block now correctly tests your functions.
    
    FILE_PATH = "../BS1.csv" # Make sure this file exists
    QUERY = "Summarize the key findings from this document."

    try:
        # Call the function you defined
        answer = get_rag_answer(FILE_PATH, QUERY)
        
        print("\n--- FINAL ANSWER ---")
        print(answer)
        
        # You can test another query
        print("\n--- Testing second query ---")
        answer_2 = get_rag_answer(FILE_PATH, "What is this data about?")
        print("\n--- FINAL ANSWER 2 ---")
        print(answer_2)

    except FileNotFoundError:
        print(f"ERROR: The file '{FILE_PATH}' was not found.")
        print("Please create the file with some dummy data.")
    except KeyError as e:
        print(f"ERROR: Missing environment variable: {e}")
        print("Please make sure your .env file has GOOGLE_API_KEY")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

