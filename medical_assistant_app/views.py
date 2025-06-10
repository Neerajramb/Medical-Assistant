# medical_assistant_app/views.py

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .llm_rag import get_rag_response # Import your RAG function

def index(request):
    """Renders the main chat interface HTML page."""
    return render(request, 'medical_assistant_app/index.html')

@csrf_exempt # Use this decorator for API views that receive POST requests
def chat_api(request):
    """
    Handles chat requests, processes user query through RAG, and returns LLM response.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()

            if not user_message:
                return JsonResponse({'response': 'Please enter a message.'}, status=400)

            # Get response from the RAG system
            assistant_response = get_rag_response(user_message)

            return JsonResponse({'response': assistant_response})

        except json.JSONDecodeError:
            return JsonResponse({'response': 'Invalid JSON in request body.'}, status=400)
        except Exception as e:
            print(f"Error in chat_api view: {e}")
            return JsonResponse({'response': 'An error occurred while processing your request.'}, status=500)
    else:
        return JsonResponse({'response': 'Only POST requests are allowed.'}, status=405)