# medical_assistant_app/llm_rag.py

import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import os
from dotenv import load_dotenv

# --- Configuration ---

# Load environment variables from .env file at the very beginning
load_dotenv()

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
COLLECTION_NAME = 'medical_knowledge'
MODEL_NAME = 'all-MiniLM-L6-v2' # A good general-purpose embedding model

# LLM API configuration (using Gemini 2.0 Flash)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Threshold for retrieved context. If total characters of retrieved docs is below this,
# or if no documents are retrieved, we fall back to direct LLM call.
MIN_CONTEXT_LENGTH = 50 # Adjust based on your data and desired behavior

# Initialize ChromaDB client and embedding model globally to avoid re-initialization
_chroma_client = None
_embedding_model = None
_chroma_collection = None

def _initialize_rag_components():
    """Initializes ChromaDB client and embedding model if not already initialized."""
    global _chroma_client, _embedding_model, _chroma_collection
    if _chroma_client is None:
        try:
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            _chroma_collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
            print(f"ChromaDB client and collection '{COLLECTION_NAME}' initialized.")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            _chroma_client = None
            _chroma_collection = None
            return False

    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(MODEL_NAME)
            print(f"Embedding model '{MODEL_NAME}' loaded.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            _embedding_model = None
            return False
    return True

def _call_gemini_api(prompt_text: str) -> str:
    """
    Helper function to send a prompt to the Gemini API and return the response.
    """
    if not GEMINI_API_KEY:
        return "LLM API key is not configured. Please set GEMINI_API_KEY in your .env file or system environment variables."

    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt_text}]})
    payload = {"contents": chat_history}

    headers = {'Content-Type': 'application/json'}
    api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    print("Sending request to LLM...") # Keep this basic log for API calls

    try:
        response = requests.post(api_url_with_key, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get("text"):
            llm_response = result["candidates"][0]["content"]["parts"][0]["text"]
            return llm_response
        else:
            print("LLM response structure unexpected or empty candidates:", result)
            return "I apologize, but I could not get a clear response from the AI. Please try rephrasing your question."

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with LLM API: {e}")
        return f"There was an issue connecting to the AI. Please check your internet connection and API key. Error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during AI call: {e}")
        return "An internal error occurred with the AI. Please try again later."

def is_medical_query(query: str) -> bool:
    """
    Uses Gemini to classify if a query is primarily medical or not.
    Returns True for medical, False for non-medical, None if classification fails.
    """
    classification_prompt = f"""
    Is the following query primarily about a medical, health, or mental health topic?
    Respond with ONLY 'YES' or 'NO'.

    Query: "{query}"
    """
    print(f"Classifying query: '{query}'") # Keep this for internal logging
    response_text = _call_gemini_api(classification_prompt)
    if response_text:
        response_text_upper = response_text.strip().upper()
        if "YES" in response_text_upper:
            return True
        elif "NO" in response_text_upper:
            return False
    print("Failed to classify query, defaulting to assume it might be medical.") # Internal logging
    return None # Indicate classification failed or is ambiguous

def get_rag_response(user_query: str) -> str:
    """
    Performs Retrieval Augmented Generation (RAG) or falls back to direct LLM call.
    """
    if not _initialize_rag_components():
        return "Error: RAG components failed to initialize. Please check server logs."

    try:
        # Step 0: Classify if the query is medical
        query_is_medical = is_medical_query(user_query)

        # If query is explicitly classified as NON-MEDICAL, return the polite message immediately.
        if query_is_medical is False:
            print(f"Query '{user_query}' classified as NON-MEDICAL. Bypassing RAG and going direct to polite fallback message.") # Internal logging
            return "I apologize, but I am a medical assistant and can only provide information related to medical, health, and mental health topics. Please ask a health-related question."

        # Proceed with RAG for medical or unclassified queries (query_is_medical is True or None)
        # 1. Embed the user query
        query_embedding = _embedding_model.encode([user_query]).tolist()

        # 2. Retrieve relevant documents from ChromaDB
        results = _chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            include=['documents']
        )

        retrieved_docs = results['documents'][0] if results['documents'] else []
        context_str = "\n".join(retrieved_docs)
        print(f"Retrieved {len(retrieved_docs)} documents for query: '{user_query}'") # Internal logging

        # Determine if retrieved context is sufficient for RAG
        if not retrieved_docs or len(context_str) < MIN_CONTEXT_LENGTH:
            # If insufficient local context for a medical/ambiguous query, allow Gemini to use general medical knowledge.
            print("Insufficient local medical context from vector DB. Relying on Gemini's general medical knowledge.") # Internal logging
            prompt = f"""
            You are a helpful medical assistant. Based on your general knowledge, please provide an accurate and helpful answer to the following medical, health, or mental health question.
            If you do not have sufficient information to provide a specific answer, please state that and advise consulting a healthcare professional.
            Question: {user_query}
            Answer:
            """
            return _call_gemini_api(prompt)
        else:
            # Sufficient local medical context found, proceed with RAG.
            print("Sufficient medical context found. Proceeding with RAG.") # Internal logging
            prompt = f"""
            You are a helpful medical assistant. Use ONLY the following provided medical information to answer the user's question.
            If the answer is NOT explicitly present in the provided information, respond with: "I don't have enough specific information on that in my medical knowledge base. Please consult a healthcare professional for accurate advice, or try rephrasing your question."
            Do not make up information.

            Medical Information:
            {context_str}

            User Question:
            {user_query}

            Answer:
            """
            return _call_gemini_api(prompt)

    except Exception as e:
        print(f"An unexpected error occurred during RAG process: {e}") # Internal logging
        return "An internal error occurred. Please try again later or consult a healthcare professional."

# Example usage (for testing this module directly)
if __name__ == "__main__":
    print("Testing RAG module directly. Ensure medical_data.txt is loaded into ChromaDB.")
    print("\n--- Test 1: Query with sufficient context (should use RAG) ---")
    test_query_rag = "What are the symptoms of a common cold?"
    response_rag = get_rag_response(test_query_rag)
    print(f"\nUser: {test_query_rag}")
    print(f"Assistant: {response_rag}")

    print("\n--- Test 2: Query medical, but likely not in data (should use Gemini's general medical knowledge) ---")
    test_query_medical_general = "What is the role of Vitamin K in human body?" # Assuming this isn't in your small data but is medical
    response_medical_general = get_rag_response(test_query_medical_general)
    print(f"\nUser: {test_query_medical_general}")
    print(f"Assistant: {response_medical_general}")

    print("\n--- Test 3: Query classified as non-medical (should trigger polite non-medical response) ---")
    test_query_fallback_non_medical = "What is quantum physics?"
    response_fallback_non_medical = get_rag_response(test_query_fallback_non_medical)
    print(f"\nUser: {test_query_fallback_non_medical}")
    print(f"Assistant: {response_fallback_non_medical}")

    print("\n--- Test 4: Another medical query not in data (should use Gemini's general medical knowledge) ---")
    test_query_medical_general_2 = "Tell me about managing chronic pain."
    response_medical_general_2 = get_rag_response(test_query_medical_general_2)
    print(f"\nUser: {test_query_medical_general_2}")
    print(f"Assistant: {response_medical_general_2}")
