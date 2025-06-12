import os

# Define the content for your README.md
readme_content = """# ⚕️ LLM-Based Medical Assistant System

An intelligent, AI-powered conversational assistant designed to provide accurate and contextually relevant information on medical, health, and mental health topics. This project showcases the power of Retrieval Augmented Generation (RAG) by combining Large Language Models (LLMs) with a specialized knowledge base, ensuring reliable and grounded responses.

## ✨ Features

* **Intelligent Query Routing:** Automatically classifies user queries as medical or non-medical to provide the most appropriate response.

* **Contextual RAG Pipeline:** Leverages a local vector database to retrieve specific medical information, augmenting LLM responses for factual accuracy and reducing hallucinations.

* **General Medical Knowledge Fallback:** If local data is insufficient for a medical query, the system intelligently defaults to the LLM's broader medical knowledge.

* **Polite Non-Medical Redirection:** Graciously informs users when queries fall outside its specialized medical scope.

* **Dynamic & Extensible Knowledge Base:** Easily update and expand the medical knowledge by modifying a simple text file, without needing to retrain the core LLM.

* **Modern Web Interface:** A responsive and intuitive chat UI built with HTML, CSS, and JavaScript.

* **Scalable Architecture:** Built on Python and Django, designed for potential expansion and production deployment.

## 🚀 Why This Project Matters (Importance & Industrial Uses)

In today's information-rich world, access to reliable health information is critical. This project addresses several key challenges and offers significant industrial value:

* **Enhanced Information Accessibility:** Provides an easy-to-use platform for individuals to quickly access general health knowledge, reducing the burden on healthcare professionals for routine queries.

* **Improved Accuracy & Trust:** By grounding LLM responses with RAG, the system significantly minimizes the risk of misinformation and hallucinations, which is paramount in sensitive domains like healthcare.

* **Scalable Knowledge Management:** The extensible knowledge base design means new research, guidelines, or medical updates can be integrated rapidly, keeping the assistant's information current without costly model retraining.

* **Operational Efficiency for Healthcare:** Can serve as a preliminary information source, triage tool, or educational resource in various healthcare settings, freeing up human experts for more complex tasks.

* **Foundation for Specialized AI:** The modular architecture provides a robust blueprint for developing highly specialized AI assistants in specific medical fields (e.g., drug interactions, rare disease information, clinical trial support).

* **Showcase of Advanced AI Techniques:** Demonstrates practical application of cutting-edge LLM, Vector Database, and RAG technologies, offering a valuable learning and development resource.

## 💻 Technologies Used

* **Python:** The core programming language.

* **Django:** High-level Python web framework for the backend.

* **Large Language Model (LLM):** Gemini 2.0 Flash API for intelligent response generation and query classification.

* **Vector Database:** [ChromaDB](https://www.trychroma.com/) for efficient semantic search and knowledge retrieval.

* **Text Embeddings:** [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) for converting text into vector representations.

* **Frontend:** HTML, Custom CSS (mimicking Tailwind), and JavaScript for a responsive and aesthetic user interface.

* **.env:** For secure management of API keys.

## 📂 Project Structure

```text
medical_assistant_project/
├── venv/                     # Python Virtual Environment
├── medical_assistant_project/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py           # Django project settings
│   ├── urls.py               # Django project URL configurations
│   └── wsgi.py
├── medical_assistant_app/    # Django application
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── views.py              # Handles web requests and calls LLM logic
│   ├── urls.py               # App-specific URL configurations
│   ├── templates/
│   │   └── medical_assistant_app/
│   │       └── index.html    # Frontend HTML template
│   └── static/
│       ├── css/
│       │   └── style.css     # Frontend CSS for styling
│       └── js/
│           └── main.js       # Frontend JavaScript for interactivity
├── medical_data.txt          # Your curated medical knowledge base (text format)
├── load_data_to_vectordb.py  # Script to process medical_data.txt into ChromaDB
├── generate_medical_facts.py # (Optional) Script to generate more facts using LLM
├── llm_rag.py                # Core RAG and LLM interaction logic
├── .env                      # Environment variables (e.g., GEMINI_API_KEY) - IMPORTANT: Add to .gitignore!
├── .gitignore                # Specifies files/directories to ignore in Git
└── manage.py                 # Django's command-line utility

```

## ⚙️ How to Run

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/YourUsername/medical-assistant-project.git](https://github.com/YourUsername/medical-assistant-project.git)
    cd medical-assistant-project

    ```

2.  **Create and Activate Virtual Environment:**

    ```bash
    python -m venv venv
    # On Windows:
    .\\venv\\Scripts\\activate
    # On macOS/Linux:
    source venv/bin/activate

    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt # (Assuming you'll create a requirements.txt)
    # If not, install individually:
    # pip install django chromadb sentence-transformers requests python-dotenv numpy

    ```

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the root of your project (same directory as `manage.py` and `medical_data.txt`).
    Add your Gemini API Key:

    ```
    GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"

    ```

    **Replace `"YOUR_ACTUAL_GEMINI_API_KEY"` with your actual API key obtained from [Google AI Studio](https://ai.google.dev/sa/api_key).**

5.  **Prepare Medical Knowledge Base:**
    Ensure you have `medical_data.txt` in the root directory with your desired medical facts.
    Run the script to populate your vector database:

    ```
    python load_data_to_vectordb.py

    ```

    (Optional: If you want to generate more data, run `python generate_medical_facts.py`)

6.  **Run Django Migrations:**

    ```
    python manage.py makemigrations medical_assistant_app
    python manage.py migrate

    ```

7.  **Start the Django Development Server:**

    ```
    python manage.py runserver

    ```

8.  **Access the Application:**
    Open your web browser and navigate to `http://127.0.0.1:8000/`.

## 📈 Future Enhancements

* **User Authentication & Profiles:** Allow users to create accounts and save chat history.

* **Advanced Data Ingestion:** Implement tools for ingesting PDFs, web pages, and other document formats into the knowledge base.

* **Source Citation:** Display the source documents from the vector database that were used to generate a response.

* **Voice Input/Output:** Integrate speech-to-text and text-to-speech capabilities.

* **Multi-modal Inputs:** Allow users to upload images (e.g., rash photos) for analysis (requires a multi-modal LLM).

* **Real-time Streaming:** Implement server-sent events (SSE) or WebSockets for streaming LLM responses.

* **Admin Panel for Knowledge Base:** Create a Django admin interface for managing `medical_data.txt` chunks directly.

* **Deployment to Cloud:** Prepare for scalable deployment on platforms like Google Cloud, AWS, or Heroku.

---

**Disclaimer:** This LLM-Based Medical Assistant System is for informational and educational purposes only and does not constitute professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns.
"""

# Define the file path in the project root
file_path = "README.md"

try:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Successfully created '{file_path}' with the correct content.")
    print("Now you can add and commit this file to your GitHub repository.")
except IOError as e:
    print(f"Error writing to file '{file_path}': {e}")
