"""
RAG-based Insurance Customer Support Chatbot.
Handles user queries related to insurance using a retrieval-augmented generation approach.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM

from vectordb import VectorDatabase


# Load environment variables
load_dotenv()


class NebiusChatLLM(BaseLLM):
    """Implementation of Nebius Chat LLM using OpenAI SDK compatibility."""
    
    model_name: str = "google/gemma-2-9b-it"  # Define as class variable
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-9b-it",
        temperature: float = 0.5,
        max_tokens: int = 500,
        top_p: float = 0.9,
        top_k: int = 50,
        api_key: Optional[str] = None,
        base_url: str = "https://api.studio.nebius.com/v1/",
        **kwargs
    ):
        """Initialize Nebius Chat LLM.
        
        Args:
            model_name: The model to use
            temperature: Temperature for text generation
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            api_key: The API key to use for Nebius API
            base_url: The base URL for Nebius API
        """
        super().__init__(**kwargs)
        self.model_name = model_name  # Set instance attribute for model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        # Use NEBIUS_API_KEY from environment if not provided
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("Nebius API key is not provided and not found in environment variables")
        
        # Initialize the OpenAI client with Nebius configuration
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call to Nebius Chat API.
        
        Args:
            prompt: The prompt to send
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        # Prepare messages for chat completion
        messages = [{"role": "user", "content": prompt}]
        
        # Prepare optional parameters
        optional_params = {}
        if stop:
            optional_params["stop"] = stop
        
        # Call the API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body={
                "top_k": self.top_k
            },
            **optional_params
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content
    
    # Implementing required abstract methods for BaseLLM
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "nebius-chat"
    
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> Any:
        """Generate text from prompts."""
        results = []
        for prompt in prompts:
            response = self._call(prompt, stop, **kwargs)
            results.append(response)
        
        # Format the output to match BaseLLM's expected structure
        from langchain.schema import LLMResult, Generation
        generations = [[Generation(text=res)] for res in results]
        return LLMResult(generations=generations)


class InsuranceChatbot:
    def __init__(
        self, 
        model_name: str = "google/gemma-2-9b-it",
        temperature: float = 0.2,
        max_tokens: int = 500,
        vector_index_name: str = "insurance_index",
        top_k: int = 5,
        use_nebius: bool = True
    ):
        """
        Initialize the insurance chatbot.
        
        Args:
            model_name: Model to use for chat completions
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            vector_index_name: Name of the FAISS index to use
            top_k: Number of documents to retrieve for each query
            use_nebius: Whether to use Nebius AI for chat completions
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        
        # Initialize the language model
        if use_nebius:
            self.llm = NebiusChatLLM(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        # Initialize vector database
        self.vector_db = VectorDatabase(use_nebius=use_nebius)
        if not self.vector_db.load_index(vector_index_name):
            print(f"Warning: Could not load vector index '{vector_index_name}'")
        
        # Create the prompt template for insurance customer support
        self._create_prompt_template()
        
        # Create LLM chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _create_prompt_template(self) -> None:
        """Create the prompt template for the chatbot."""
        template = """You are a helpful insurance customer support assistant. 
Your task is to provide accurate and helpful information about insurance policies, claims, and general insurance questions.
Only respond to insurance-related queries. If the question is not related to insurance, politely inform the user that you can only assist with insurance-related matters.

Use the following retrieved information to answer the user's question. If you don't know the answer based on the retrieved information, acknowledge that you don't have that specific information but provide general guidance if possible.

Context information from knowledge base:
{context}

User question: {question}

Your response (be concise, professional, and helpful):
"""
        
        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _retrieve_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for the query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents
        """
        if not self.vector_db.vector_store:
            print("Warning: Vector store not initialized")
            return []
        
        return self.vector_db.similarity_search(query, k=self.top_k)
    
    def _format_documents_as_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents as context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', 'N/A')
            
            context_part = f"Document {i} (Source: {source}"
            if page != 'N/A':
                context_part += f", Page: {page}"
            context_part += f"):\n{doc.page_content}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def is_insurance_related(self, query: str) -> bool:
        """
        Determine if a query is related to insurance.
        This is a simple keyword-based check, but could be improved with a classifier.
        
        Args:
            query: User query
            
        Returns:
            True if the query is likely insurance-related
        """
        insurance_keywords = [
            "insurance", "policy", "claim", "premium", "coverage", "deductible",
            "benefits", "insured", "policyholder", "underwriting", "risk",
            "liability", "accident", "damage", "loss", "health", "life",
            "auto", "car", "vehicle", "property", "home", "medical", "dental",
            "vision", "disability", "term", "whole life", "reimbursement"
        ]
        
        query_lower = query.lower()
        
        # Check if any insurance keyword is in the query
        return any(keyword in query_lower for keyword in insurance_keywords)
    
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Generate an answer for the user query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer and metadata
        """
        # Check if the query is insurance-related
        if not self.is_insurance_related(query):
            return {
                "answer": "I'm an insurance customer support assistant. I can only assist with insurance-related questions. How can I help you with your insurance needs?",
                "sources": [],
                "is_insurance_related": False
            }
        
        # Retrieve relevant documents
        documents = self._retrieve_relevant_documents(query)
        
        # Format documents as context
        context = self._format_documents_as_context(documents)
        
        # Generate answer using RAG
        response = self.chain.invoke({
            "context": context,
            "question": query
        })
        
        # Prepare source information
        sources = []
        for doc in documents:
            source = {
                "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source)
        
        return {
            "answer": response["text"],
            "sources": sources,
            "is_insurance_related": True
        }


if __name__ == "__main__":
    # Example usage
    chatbot = InsuranceChatbot()
    response = chatbot.answer("How do I file a claim for car damage?")
    print(response["answer"])
