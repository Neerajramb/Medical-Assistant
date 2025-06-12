# medical_assistant_app/llm_rag.py

import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

try:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    CHROMA_DB_PATH = os.path.join(SCRIPT_DIR, '..', 'chroma_db')
except NameError:
    CHROMA_DB_PATH = os.path.join(os.getcwd(), 'chroma_db')
    print(f"Warning: '__file__' not defined. Using CWD for ChromaDB path: {CHROMA_DB_PATH}")


COLLECTION_NAME = 'medical_knowledge'
MODEL_NAME = 'all-MiniLM-L6-v2'
N_RESULTS = 3

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# --- Global Component Initialization ---
# (This section is correct and remains unchanged)
_chroma_client = None
_embedding_model = None
_chroma_collection = None

def _initialize_rag_components():
    """Initializes ChromaDB client and embedding model if not already initialized."""
    global _chroma_client, _embedding_model, _chroma_collection
    if _chroma_client is None:
        try:
            print(f"Initializing ChromaDB client at path: {CHROMA_DB_PATH}")
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            _chroma_collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
            print(f"ChromaDB client and collection '{COLLECTION_NAME}' initialized.")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            _chroma_client = None; _chroma_collection = None
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
    """Helper function to send a prompt to the Gemini API and return the response."""
    # (This function is correct and remains unchanged)
    if not GEMINI_API_KEY:
        return "LLM API key is not configured. Please set GEMINI_API_KEY in your .env file."

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}
    headers = {'Content-Type': 'application/json'}
    api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    print("--- Sending request to LLM ---")
    
    try:
        response = requests.post(api_url_with_key, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content", {}).get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            print("LLM response structure unexpected or empty:", result)
            return "I apologize, but I received an unusual response from the AI. This might be due to a content filter. Please try rephrasing."
    except requests.exceptions.HTTPError as e:
        error_details = f"HTTP Error: {e.response.status_code}"
        try:
            error_message = e.response.json().get("error", {}).get("message", "No details.")
            error_details += f" - {error_message}"
        except json.JSONDecodeError:
            pass
        print(f"Error communicating with LLM API: {error_details}")
        return "There was an issue connecting to the AI. Please check the API key and model name."
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with LLM API: {e}")
        return f"There was an issue connecting to the AI. Error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during AI call: {e}")
        return "An internal error occurred with the AI. Please try again later."


def get_rag_response(user_query: str) -> str:
    """
    Handles all user queries by building a single, intelligent prompt that instructs
    the LLM to prioritize local context but seamlessly fall back to general knowledge.
    """
    if not _initialize_rag_components():
        return "Error: RAG components failed to initialize. Please check server logs."

    try:
        # Step 1: Always retrieve context to inform the LLM.
        query_embedding = _embedding_model.encode([user_query]).tolist()
        results = _chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=N_RESULTS,
            include=['documents']
        )
        retrieved_docs = results['documents'][0] if results.get('documents') else []
        context_str = "\n".join(retrieved_docs)
        
        if retrieved_docs:
            print(f"Retrieved {len(retrieved_docs)} documents for query: '{user_query}'")
        else:
            print(f"No relevant documents found for query: '{user_query}'. Will rely on general knowledge.")

        # <<< CORRECTION: The prompt is refined to be extremely direct about the fallback, preventing "I don't know" responses. >>>
        prompt = f"""### Persona
You are a knowledgeable, friendly, and helpful medical information assistant.

### Core Task
Your goal is to answer the user's message accurately by following these rules in order.

### Rules of Engagement
1.  **Greeting**: If the user provides a simple greeting or chitchat (e.g., 'hello', 'thank you'), respond warmly and naturally, then invite them to ask a health-related question.

2.  **Off-Topic**: If the user asks a question that is clearly NOT related to medicine, health, or wellness, you MUST politely state your purpose. Respond with: "I apologize, but as a medical information assistant, I can only provide information related to health topics. How can I help you with a health question?"

3.  **Medical Question**: If the user asks a medical question (from simple symptoms to technical codes), you MUST follow this process:
    a. **Prioritize Provided Information:** First, check if the "Provided Medical Information" below contains a relevant answer to the user's question. If it does, use it to construct your answer.
    b. **Seamless Fallback:** If the "Provided Medical Information" is empty or does not answer the question, you MUST immediately use your own general knowledge to provide a complete and accurate answer. **Never state that you couldn't find it in your database.** Simply proceed to answer.
    c. **Disclaimer:** Always end any medical-related answer with this disclaimer: "Please remember, this information is for educational purposes only and is not a substitute for professional medical advice."

### Provided Medical Information
{context_str}

### User's Message
"{user_query}"

### Your Answer:"""
        
        # Step 3: Call the LLM with the single, powerful prompt
        return _call_gemini_api(prompt)

    except Exception as e:
        print(f"An unexpected error occurred during RAG process: {e}")
        return "An internal error occurred. Please try again later."

# Example usage (for testing this module directly)
if __name__ == "__main__":
    print("Testing RAG module directly. This version ensures seamless fallback without 'I don't know' messages.")

    print("\n--- Test 1: Medical Question in the database (should use RAG) ---")
    response = get_rag_response("What are the symptoms of a common cold?")
    print(f"User: What are the symptoms of a common cold?\nAssistant: {response}")

    print("\n--- Test 2: Medical Question NOT in database (should use Gemini's knowledge seamlessly) ---")
    response = get_rag_response("What is the average recovery time for an ACL surgery?")
    print(f"User: What is the average recovery time for an ACL surgery?\nAssistant: {response}")

    print("\n--- Test 3: Technical Medical Question NOT in database (should use Gemini's knowledge seamlessly) ---")
    response = get_rag_response("What is the ICD-10 code for type 2 diabetes without complications?")
    print(f"User: What is the ICD-10 code for type 2 diabetes without complications?\nAssistant: {response}")

    print("\n--- Test 4: Off-topic question (should politely decline) ---")
    response = get_rag_response("What's a good recipe for lasagna?")
    print(f"User: What's a good recipe for lasagna?\nAssistant: {response}")