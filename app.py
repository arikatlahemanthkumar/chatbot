from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logging
import time
import os
import re

from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import Graph, END
from langchain_together import Together
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app with lifespan hook
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting chatbot application...")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Gemini AI Chatbot", lifespan=lifespan)

# Mount static files (e.g., HTML frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangChain components
def create_langchain_components():
    try:
        llm = Together(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            temperature=0.2,
            together_api_key=os.getenv("TOGETHER_API_KEY"),
            
        )
       

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        """ prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful and knowledgeable AI assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])

         chain = RunnableSequence(
            system_message=SystemMessage(content="You are a helpful and knowledgeable AI assistant."),
            messages_placeholder=MessagesPlaceholder(variable_name="chat_history"),
            human_message=HumanMessage(content="{input}"),
            llm=llm,
            memory=memory
        ) """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful and knowledgeable AI assistant. Your responses should be:

1. Direct and concise
2. Factually accurate
3. Helpful and practical
4. Free from unnecessary explanations or meta-commentary

Focus on providing the most relevant information or solution to the user's query. If you're unsure about something, simply say you don't know rather than speculating.

Example responses:
User: "What's the capital of France?"
Assistant: "Paris"

User: "How do I make a cup of tea?"
Assistant: "1. Boil water\n2. Add tea leaves to a cup\n3. Pour hot water\n4. Steep for 3-5 minutes\n5. Remove leaves and enjoy"

Keep your responses focused and to the point."""),
            ("human", "{input}")
        ])

        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            output_parser=StrOutputParser() 
        )
        def process_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = chain.invoke({"input": input_data["input"]})
                response = result["text"].strip()
                # Remove any content between <think> tags and the tags themselves
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                # Remove any standalone <think> or </think> tags
                response = response.replace('<think>', '').replace('</think>', '')
                return {"text": response}
            except Exception as e:
                logger.error(f"LangChain error: {str(e)}")
                return {"text": "Sorry, I couldn't process that message."}

        return process_input, memory
    except Exception as e:
        logger.error(f"Failed to initialize LangChain: {str(e)}")
        raise

# Create LangGraph workflow
def create_langgraph_workflow(process_input):
    def process_input_node(state):
        return {"input": state["input"]}

    def generate_response_node(state):
        response = process_input({"input": state["input"]})
        return {"response": response["text"]}

    graph = Graph()
    graph.add_node("process_input", process_input_node)
    graph.add_node("generate_response", generate_response_node)

    graph.add_edge("process_input", "generate_response")
    graph.add_edge("generate_response", END)

    graph.set_entry_point("process_input")
    return graph.compile()

# Initialize LangChain + LangGraph
try:
    process_input, memory = create_langchain_components()
    workflow = create_langgraph_workflow(process_input)
    logger.info("LangChain and LangGraph initialized successfully")
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise

# Pydantic models
class ChatMessage(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=1000)

class ChatResponse(BaseModel):
    response: str

class HealthResponse(BaseModel):
    status: str
    langchain: str
    timestamp: float

# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        _ = process_input({"input": "Hello"})
        langchain_status = "healthy"
    except Exception as e:
        langchain_status = f"unhealthy: {str(e)}"

    return HealthResponse(
        status="ok",
        langchain=langchain_status,
        timestamp=time.time()
    )

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        logger.info(f"Received from {message.user_id}: {message.message}")
        result = workflow.invoke({"input": message.message, "user_id": message.user_id})
        logger.info(f"Response: {result['response']}")
        return ChatResponse(response=result["response"])
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Root route for UI
@app.get("/")
async def get_chat_interface():
    return FileResponse("static/index.html")

# Run with Uvicorn
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    logger.info(f"Running on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
