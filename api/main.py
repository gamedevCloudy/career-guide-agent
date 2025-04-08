from fastapi import FastAPI, Request, Response, Depends, Cookie, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uuid
from typing import Optional, List, Dict, Any
import os
import json
from pathlib import Path

# Import your agent functions
from api.agents.runner import chat_with_agent
from api.agents.utils import AgentState

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request, thread_id: Optional[str] = Cookie(None)):
    """Render the chat interface."""
    # Generate a new thread_id if none exists
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "thread_id": thread_id}
    )

@app.post("/api/chat")
async def process_chat(
    message: str = Form(...),
    thread_id: Optional[str] = Cookie(None),
    response: Response = None
):
    """Process chat messages and return responses."""
    # Create a new thread_id if none exists
    if not thread_id:
        thread_id = str(uuid.uuid4())
        response.set_cookie(key="thread_id", value=thread_id, max_age=3600*24*7)  # 1 week expiry
    
    print(f"Processing message: '{message}' for thread: {thread_id}")
    
    try:
        # Process the message through the graph
        final_state = chat_with_agent(message, thread_id)
        
        # Get the last assistant message as the response
        messages = final_state.get("messages", [])
        assistant_messages = [msg for msg in messages 
                              if hasattr(msg, "type") and msg.type == "ai"]
        
        if assistant_messages:
            # Get the most recent assistant message
            last_message = assistant_messages[-1]
            
            # Prepare the response
            response_data = {
                "content": last_message.content,
                "name": getattr(last_message, "name", "Assistant")
            }
            
            print(f"Responding with: {response_data['name']}: '{response_data['content'][:50]}...'")
            return JSONResponse(content=response_data)
        else:
            # Fallback if no assistant message found
            return JSONResponse(
                content={"content": "I'm processing your request...", "name": "System"}
            )
            
    except Exception as e:
        import traceback
        print(f"Error processing message: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            content={"content": f"Sorry, I encountered an error: {str(e)}", "name": "System"},
            status_code=500
        )

@app.get("/api/conversation")
async def get_conversation(thread_id: Optional[str] = Cookie(None)):
    """Get the full conversation history for debugging purposes."""
    if not thread_id:
        return JSONResponse(content={"error": "No thread_id provided"}, status_code=400)
    
    try:
        # This is just for debugging - you might want to remove this in production
        from api.agents.runner import create_career_optimization_graph
        graph = create_career_optimization_graph()
        config = {'configurable': {'thread_id': thread_id}}
        
        state = graph.get_state(config)
        
        # Convert messages to a serializable format
        messages = []
        for msg in state.get("messages", []):
            message_data = {
                "role": "user" if hasattr(msg, "type") and msg.type == "human" else "assistant",
                "content": msg.content
            }
            
            if hasattr(msg, "name") and msg.name:
                message_data["name"] = msg.name
                
            messages.append(message_data)
        
        return JSONResponse(content={"messages": messages})
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to retrieve conversation: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)