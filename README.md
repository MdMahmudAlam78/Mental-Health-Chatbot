# Mental Health Chatbot using RAG

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot designed to answer mental health queries using WHO's Mental Health Gap Guidelines PDF as the knowledge base. It leverages LangChain, FAISS, and transformer embeddings to provide accurate and reliable information.

## Features
- PDF document loading and chunking
- Semantic search with vector embeddings
- Question-answering with pre-trained language models
- Simple Flask/Gradio interface for interaction

## Folder Structure
- `app.py`: Main application code
- `run_app.py`: Script to launch the chatbot
- `build_index.py`: Script to create vector index from PDF
- `faiss_index/`: Stores vector index files
- `mental_health_book.pdf`: WHO guidelines document used as source
- `requirements.txt`: List of required Python packages

## Setup Instructions
1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd mental_health_chatbot
2.Create virtual environment and activate:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
  ```
   
## Future Work: Multi-Turn Mental Health Chatbot

This project is currently focused on single-turn interactions using the RAG model. Future improvements include:

- **Multi-turn conversation management:**  
  Enabling the chatbot to remember and contextually process multiple turns in a conversation for more natural and effective interactions.

- **Enhanced dialogue state tracking:**  
  Tracking user intent, sentiment, and conversation context over multiple exchanges.

- **Integration of NLP-based diagnostic models:**  
  Combining retrieval with fine-tuned NLP models to provide more precise mental health diagnosis and recommendations.

- **User personalization:**  
  Tailoring responses based on user history and preferences while ensuring privacy and data security.

- **Deployment improvements:**  
  Building a scalable and responsive web or mobile interface for real-time usage.

---

## Contributions & Contact
Contributions are welcome! Feel free to open issues or submit pull requests.

For questions or collaboration, contact: [mahmudalam3210@gmail.com]

