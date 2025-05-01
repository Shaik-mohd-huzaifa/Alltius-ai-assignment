# Insurance Customer Support Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot for insurance customer support using Python, OpenAI/Nebius, and FAISS vector database.

## Features

- Web scraping to ingest insurance and financial content from websites
- Document processing for insurance policy documents and manuals
- FAISS vector database for efficient similarity search
- RAG-based chatbot that provides accurate, concise responses
- Specialized for insurance-related queries
- REST API for integrating with the React frontend
- Modern, responsive web interface

## Project Structure

```
Alltius.ai Assignment/
├── insurance-chatbot/        # Backend application
│   ├── data/                 # Directory for storing raw documents and web data
│   │   ├── documents/        # Insurance policy documents (PDFs)
│   │   └── webpage.json      # List of URLs to scrape
│   ├── vectorstore/          # Directory for storing the FAISS index
│   ├── src/
│   │   ├── web_scraper.py    # For scraping websites
│   │   ├── document_loader.py # For loading and processing documents
│   │   ├── vectordb.py       # For setting up the FAISS vector DB
│   │   ├── rag_engine.py     # For the RAG implementation
│   │   └── api.py            # For the Flask API server
│   ├── create_embeddings.py  # Script to create embeddings and build vector store
│   ├── start_api.py          # Script to start the API server
│   ├── requirements.txt      # Dependencies
│   └── README.md             # Project documentation
├── frontend/                 # React frontend application
└── README.md                 # This file - main project documentation
```

## Prerequisites

- Python 3.8 or higher
- Node.js and npm (for the frontend)
- Nebius API key or OpenAI API key

## Installation

### Backend Setup

1. Clone this repository or download it to your local machine
2. Navigate to the insurance-chatbot directory:

```bash
cd insurance-chatbot
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the insurance-chatbot directory with your API key:

```
# For Nebius API
NEBIUS_API_KEY=your_nebius_api_key_here

# For OpenAI (if using OpenAI instead)
# OPENAI_API_KEY=your_openai_api_key_here
```

### Frontend Setup

1. Navigate to the frontend directory from the root of the project:

```bash
cd frontend
```

2. Install the dependencies:

```bash
npm install
```

## Configuration

### Using OpenAI Instead of Nebius

By default, the application is configured to use Nebius API. If you want to use OpenAI instead:

1. In the `.env` file inside the insurance-chatbot directory, comment out or remove the Nebius API key and uncomment/add the OpenAI API key:

```
# NEBIUS_API_KEY=your_nebius_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

2. In `insurance-chatbot/src/rag_engine.py`, modify the OpenAI client initialization:

```python
# Replace this:
self.client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# With this:
self.client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
```

3. Update the model names to use OpenAI models:

```python
# Replace these:
self.model = "google/gemma-2-9b-it"
self.embedding_model = "BAAI/bge-multilingual-gemma2"

# With these:
self.model = "gpt-3.5-turbo"  # or "gpt-4" or any other OpenAI chat model
self.embedding_model = "text-embedding-ada-002"  # or any other OpenAI embedding model
```

## Usage

### Creating the Vector Database

Before using the chatbot, you need to process documents and webpages to build the knowledge base:

1. Put your insurance PDF documents in the `insurance-chatbot/data/documents` directory
2. Edit `insurance-chatbot/data/webpage.json` to include the URLs you want to scrape:

```json
[
  "https://www.angelone.in/support/add-and-withdraw-funds/add-funds",
  "https://www.angelone.in/support/account-opening/account-opening"
]
```

3. Run the embedding creation script from the insurance-chatbot directory:

```bash
cd insurance-chatbot
python create_embeddings.py
```

This will:
- Extract text from PDF documents in the `data/documents` directory
- Scrape content from the URLs in `data/webpage.json`
- Split the content into chunks
- Generate embeddings for each chunk
- Store everything in the FAISS vector database in the `vectorstore` directory

### Starting the Backend Server

To start the API server:

```bash
cd insurance-chatbot
python start_api.py
```

The server will run on `http://localhost:5000`.

### Starting the Frontend Application

To start the frontend development server:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in your console).

## API Reference

### Chat Endpoint

- **URL**: `/api/chat`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "query": "How do I file a claim?",
    "session_id": "optional-session-id",
    "use_rag": true
  }
  ```
- **Response**:
  ```json
  {
    "answer": "To file a claim, you need to...",
    "is_insurance_related": true,
    "sources": [
      "https://www.angelone.in/support/add-and-withdraw-funds/add-funds"
    ],
    "session_id": "optional-session-id",
    "timestamp": 1619712345.6789,
    "using_rag": true
  }
  ```

### Health Check Endpoint

- **URL**: `/api/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "ok",
    "timestamp": 1619712345.6789,
    "chatbot_initialized": true,
    "rag_enabled": true,
    "vector_index_size": 267
  }
  ```

## Customizing the System Prompt

The system prompt controls how the LLM generates responses. You can modify it in `src/rag_engine.py`:

```python
messages = [
    {
        "role": "system", 
        "content": "You are an insurance assistant providing extremely concise..."
    },
    # ...
]
```

Adjust the prompt to change the style, tone, or level of detail in the responses.

## Troubleshooting

### Vector Database Issues

If you encounter problems with the vector database:

1. Delete the contents of the `vectorstore` directory:
   ```bash
   rm -rf vectorstore/*
   ```

2. Run the embedding creation script again:
   ```bash
   python create_embeddings.py
   ```

### API Connection Issues

If the frontend can't connect to the API:

1. Make sure the API server is running (`python start_api.py`)
2. Check that CORS is properly configured in `src/api.py`
3. Verify that the frontend is using the correct API URL (`http://localhost:5000`)

## Example Queries

The chatbot is designed to answer insurance and financial questions such as:
- "How do I add funds to my account?"
- "What is the process for withdrawing money?"
- "How do quarterly settlements work?"
- "What are the charges for different services?"

## Customizing the Frontend

The frontend is built with React and can be customized by editing the files in the `frontend` directory:

- `src/App.jsx`: Main application component
- `src/App.css`: Styling for the application

## License

MIT
