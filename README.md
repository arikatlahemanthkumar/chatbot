# Chatbot with LangGraph, LangChain, and Gemini AI

This is a chatbot application that uses LangGraph, LangChain, and Gemini AI for natural language processing, with Neon DB for persistent storage.

## Features

- Integration with Google's Gemini AI for natural language processing
- Persistent chat history storage using Neon DB
- RESTful API using FastAPI
- User-specific chat history retrieval

## Prerequisites

- Python 3.8 or higher
- A Gemini AI API key
- A Neon DB connection string

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

5. Edit the `.env` file with your actual credentials:
   - GEMINI_API_KEY: Your Google Gemini AI API key
   - NEON_DATABASE_URL: Your Neon DB connection string

## Running the Application

Start the server:
```bash
python app.py
```

The server will start on `http://localhost:8000` by default.

## API Endpoints

### Chat
- **POST** `/chat`
  - Request body:
    ```json
    {
        "user_id": "user123",
        "message": "Hello, how are you?"
    }
    ```
  - Response:
    ```json
    {
        "response": "I'm doing well, thank you for asking!"
    }
    ```

### Chat History
- **GET** `/history/{user_id}`
  - Returns the chat history for the specified user
  - Response:
    ```json
    [
        {
            "message": "Hello",
            "response": "Hi there!",
            "timestamp": "2024-01-01T12:00:00"
        }
    ]
    ```

## Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

MIT 