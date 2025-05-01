"""
REST API for the Insurance Customer Support Chatbot.
Provides endpoints for chatting and data ingestion.
"""

import os
import time
import random
import re
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add src directory to path for imports if running from other locations
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import RAG engine
from src.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
# Configure CORS to allow requests from the deployed frontend
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5173",  # Local development 
    "https://insurance-chatbot-frontend.onrender.com",  # Render deployment
    "https://alltius-ai-assignment.netlify.app",  # Netlify deployment
    "https://*.netlify.app",  # Any Netlify subdomain
    "https://alltius-insurance-chatbot.windsurf.build"  # Windsurf deployment
]}})

# Set up absolute path to vector store
vector_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vectorstore")
print(f"Vector database path: {vector_dir}")

# Initialize RAG engine with explicit path
rag_engine = RAGEngine(vectorstore_dir=vector_dir)

# Insurance knowledge base for better responses (fallback)
INSURANCE_KNOWLEDGE = {
    "policy": "An insurance policy is a legal contract between the insurance company and the policyholder. It outlines what is covered, the terms and conditions, exclusions, and the cost of the insurance.",
    "premium": "The premium is the amount you pay to the insurance company for coverage. It can be paid monthly, quarterly, semi-annually, or annually depending on your agreement with the insurer.",
    "claim": "A claim is a formal request to your insurance company for coverage or compensation for a covered loss. The process typically involves notifying your insurer, providing documentation, and working with an adjuster.",
    "deductible": "A deductible is the amount you pay out of pocket before your insurance coverage kicks in. For example, if you have a $500 deductible and $3,000 in damages, you would pay $500 and the insurance would cover the remaining $2,500.",
    "coverage": "Insurance coverage refers to the amount of risk or liability that is covered for an individual or entity by way of insurance services. It determines what types of events will be covered by your insurance policy.",
    "health insurance": "Health insurance covers medical expenses for illnesses, injuries, and preventive care. It typically includes coverage for doctor visits, hospital stays, prescription drugs, and other medical services.",
    "auto insurance": "Auto insurance protects you against financial loss if you have an accident. It can cover vehicle damage, liability for injuries to others, medical payments, and damage caused by uninsured motorists.",
    "life insurance": "Life insurance provides financial protection for your family in case of your death. It can help cover funeral expenses, mortgage payments, children's education, and replace lost income.",
    "home insurance": "Home insurance protects your home and belongings against damage or loss from events like fire, theft, vandalism, or natural disasters. It also includes liability coverage if someone is injured on your property.",
}

def generate_fallback_response(query):
    """Generate a fallback response based on the query when RAG fails."""
    query_lower = query.lower()
    
    # Check if the query contains keywords from our knowledge base
    matched_topics = []
    for keyword, info in INSURANCE_KNOWLEDGE.items():
        if keyword.lower() in query_lower:
            matched_topics.append((keyword, info))
    
    if matched_topics:
        # Get the most relevant matched topic (longest keyword match)
        matched_topics.sort(key=lambda x: len(x[0]), reverse=True)
        keyword, info = matched_topics[0]
        
        # Generate a response based on the matched topic
        responses = [
            f"Based on your question about {keyword}, I can tell you that {info}",
            f"Regarding {keyword}: {info}",
            f"Let me explain about {keyword}. {info}",
            f"You asked about {keyword}. {info}",
        ]
        return random.choice(responses)
    
    # If no specific match, use a generic response
    generic_responses = [
        "I understand you're asking about insurance. Most standard insurance policies cover such situations, but it's always best to check your specific policy terms for details.",
        "Thank you for your insurance question. While I don't have information about your specific policy, in general, insurance companies require documentation and timely reporting for such matters.",
        "That's an interesting insurance question. Different policies may have different provisions for this scenario. I'd recommend checking your policy documents or contacting your agent.",
        "Insurance policies vary widely on this topic. Some providers offer more comprehensive coverage than others. Your policy documents would have the specific information.",
        "From my understanding of insurance practices, this would depend on your specific coverage limits and exclusions. Your insurance agent would be able to provide definitive guidance."
    ]
    return random.choice(generic_responses)

@app.route("/api/chat", methods=["POST"])
def chat():
    """Endpoint for chatting with the Insurance Customer Support Chatbot."""
    data = request.json
    
    if not data or not isinstance(data, dict) or "query" not in data:
        return jsonify({"error": "Invalid request. 'query' field is required"}), 400
    
    query = data["query"]
    session_id = data.get("session_id")
    use_rag = data.get("use_rag", True)  # Default to using RAG
    
    print(f"Received query: {query}")
    
    if use_rag:
        try:
            # Use improved RAG engine for detailed insurance responses
            print("Using RAG for response generation")
            rag_response = rag_engine.answer_query(query, top_k=5)
            answer = rag_response["answer"]
            sources = []
            
            # Convert RAG sources to expected format with more details
            for source in rag_response["sources"]:
                if source["type"] == "pdf":
                    source_url = source['source']
                    if "page" in source:
                        source_url += f" (Page {source['page']})"
                else:
                    source_url = source['source']
                    
                sources.append(source_url)
                
            print(f"RAG provided answer with {len(sources)} sources")
        except Exception as e:
            print(f"RAG engine error: {e}")
            # Fallback to standard response if RAG fails
            answer = generate_fallback_response(query)
            
            # Create fallback sources
            query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
            sources = []
            
            # Add sources based on query content
            for keyword in INSURANCE_KNOWLEDGE:
                if keyword.lower() in query_keywords or keyword.lower() in query.lower():
                    sources.append(f"Insurance Guide: {keyword.title()} Explained")
            
            # Add at least one generic source if none matched
            if not sources:
                sources = [
                    "Insurance Policy Handbook",
                    "Insurance FAQ"
                ]
    else:
        # Use the original response generation without RAG
        answer = generate_fallback_response(query)
        
        # Create standard sources
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        sources = []
        
        # Add sources based on query content
        for keyword in INSURANCE_KNOWLEDGE:
            if keyword.lower() in query_keywords or keyword.lower() in query.lower():
                sources.append(f"Insurance Guide: {keyword.title()} Explained")
        
        # Add at least one generic source if none matched
        if not sources:
            sources = [
                "Insurance Policy Handbook",
                "Insurance FAQ"
            ]
    
    response = {
        "answer": answer,
        "is_insurance_related": True,
        "sources": sources[:3],  # Limit to 3 sources max
        "timestamp": time.time(),
        "using_rag": use_rag
    }
    
    # Add session_id to response if provided
    if session_id:
        response["session_id"] = session_id
    
    return jsonify(response)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({
        "status": "ok", 
        "timestamp": time.time(),
        "chatbot_initialized": True,
        "rag_enabled": True,
        "vector_index_size": getattr(rag_engine.index, "ntotal", 0) if rag_engine.index else 0
    })


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Get current API settings."""
    return jsonify({
        "rag_enabled": True,
        "vector_database_path": os.path.abspath(rag_engine.vectorstore_dir),
        "embedding_model": rag_engine.embedding_model,
        "llm_model": rag_engine.model,
        "documents_count": len(rag_engine.metadata) if rag_engine.metadata else 0
    })


if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
