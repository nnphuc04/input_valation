import os
import pandas as pd
import requests
import json
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma  # Sửa import
from langchain_community.embeddings import HuggingFaceEmbeddings  # Sử dụng HuggingFace thay vì OpenAI
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from unstructured.partition.html import partition_html  # Giữ nguyên import này

# DeepSeek API URL
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"

# Custom LLM class cho DeepSeek
class DeepSeekLLM(LLM):
    api_key: str
    model_name: str = "deepseek-chat"
    temperature: float = 0.2
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        
        if stop:
            payload["stop"] = stop
            
        try:
            response = requests.post(
                f"{DEEPSEEK_API_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LOI] khi goi DeepSeek API: {e}")
            return f"Error: {str(e)}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature
        }

class RagApplication:
    def __init__(self, api_key):
        try:
            # Initialize DeepSeek LLM
            print("Khoi tao DeepSeek LLM...")
            self.llm = DeepSeekLLM(
                api_key=api_key,
                model_name="deepseek-chat",
                temperature=0.2
            )
            
            # Kiem tra API key hop le
            try:
                self.llm._call("Test connection")
                print("[OK] Ket noi DeepSeek API thanh cong!")
            except Exception as e:
                print(f"[CANH BAO] Khong the ket noi toi DeepSeek API. Se tiep tuc nhung co the gap loi sau: {e}")
            
            # Su dung HuggingFace Embeddings
            print("Khoi tao Embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Text splitter for chunking documents
            self.text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Vector store path
            self.persist_directory = "db"
            
            # Initialize vector DB or load if exists
            if os.path.exists(self.persist_directory):
                self.vectordb = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print(f"[OK] Da tai vector database tu {self.persist_directory}")
            else:
                self.vectordb = None
                print("[INFO] Chua co vector database, se tao moi khi nhap tai lieu")
                
        except Exception as e:
            print(f"[LOI] Khi khoi tao: {e}")
            raise
    
    def process_table_data(self, file_path):
        """Process tabular data from file and convert to HTML using pandas and unstructured"""
        # Load the tabular data based on file type
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or Excel file.")
        
        # Convert DataFrame to HTML using pandas
        html_content = df.to_html()
        
        try:
            # Parse the HTML content using unstructured
            elements = partition_html(text=html_content)
            
            # Extract text from elements and join into a single document
            document_text = "\n".join([str(element) for element in elements])
        except Exception as e:
            # Fallback if there's an issue with unstructured
            print(f"[CANH BAO] Loi khi xu ly HTML: {e}")
            print("[INFO] Su dung noi dung HTML truc tiep")
            document_text = html_content
            
        return document_text
    
    def ingest_documents(self, document_text):
        """Create embeddings and store in ChromaDB"""
        # Split documents into chunks
        chunks = self.text_splitter.split_text(document_text)
        
        # Create vector store
        self.vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectordb.persist()
        print(f"Da nhap {len(chunks)} phan tai lieu vao co so du lieu vector.")
    
    def create_qa_chain(self):
        """Create a retrieval QA chain"""
        if not self.vectordb:
            raise ValueError("Vector database is not initialized. Please ingest documents first.")
        
        # Create retriever
        retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain
    
    def query(self, question):
        """Query the RAG system with a question"""
        qa_chain = self.create_qa_chain()
        response = qa_chain({"query": question})
        
        return {
            "answer": response["result"],
            "source_documents": [doc.page_content for doc in response["source_documents"]]
        }

# Example usage
if __name__ == "__main__":
    print("=== Ung dung RAG voi DeepSeek, ChromaDB, va LangChain ===\n")
    
    try:
        # Luôn yêu cầu nhập API key thủ công
        api_key = input("Vui long nhap DeepSeek API key cua ban: ")
        
        rag_app = RagApplication(api_key)
        
        # Example: Process a tabular file
        file_path = "sample_data.csv"
        print(f"\nDang xu ly file: {file_path}")
        
        document_text = rag_app.process_table_data(file_path)
        rag_app.ingest_documents(document_text)
        
        # Query example
        print("\n=== Che do hoi dap ===")
        print("Go 'exit' de thoat")
        
        while True:
            question = input("\nNhap cau hoi cua ban: ")
            if question.lower() == 'exit':
                break
                
            response = rag_app.query(question)
            print("\nTra loi:", response["answer"])
            print("\nNguon du lieu:")
            for i, source in enumerate(response["source_documents"], 1):
                print(f"  {i}. {source[:100]}...")
    
    except Exception as e:
        print(f"\n[LOI]: {e}")