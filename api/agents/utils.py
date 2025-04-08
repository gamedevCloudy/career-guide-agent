# agents/utils.py
import os
import sqlite3
from dotenv import load_dotenv
from typing import Annotated, List, Optional, Literal, Union

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

DATABASE_URI = os.getenv('DATABASE_URI', 'agent_checkpoint.sqlite')

# Define Agent names for routing
AgentName = Literal["ProfileAnalyzer", "JobFitAnalyzer", "CareerAdvisor", "Counceller", "Supervisor"]

class AgentState(TypedDict):
    """State for the career optimization agent workflow."""
    messages: Annotated[list, add_messages]
    profile_data: Optional[List[Document]]
    next: Optional[Union[AgentName, str]]

def create_sqlite_memory():
    """Creates a SQLite-based checkpoint saver."""
    try:
        # Ensure the directory exists if DATABASE_URI includes a path
        db_dir = os.path.dirname(DATABASE_URI)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        conn = sqlite3.connect(DATABASE_URI, check_same_thread=False)
        return SqliteSaver(conn=conn)
    except Exception as e:
        print(f"Error creating SQLite memory at {DATABASE_URI}: {e}")
        print("Falling back to MemorySaver.")
        return MemorySaver()

# Using MemorySaver for development - switch to SQLite for production
memory = MemorySaver()

# System prompt for the supervisor
SUPERVISOR_SYSTEM_PROMPT = (
    "You are a supervisor tasked with managing a conversation between specialized career guidance agents.\n"
    "Given the conversation history, determine which agent should act next:\n"
    "- ProfileAnalyzer: For LinkedIn profile analysis\n"
    "- JobFitAnalyzer: For assessing job fit after profile analysis\n"
    "- CareerAdvisor: For career guidance after profile and job fit analysis\n"
    "- Counceller: For general conversation and synthesizing information\n\n"
    "Route to the appropriate agent based on the conversation context and user needs."
)

def make_agent_system_prompt(role_description: str) -> str:
    """Creates a standard system prompt for specialized agents."""
    return (
        f"You are a specialized career guidance agent. Your specific role is: {role_description}\n\n"
        "Use the provided tools to perform your task based on the current conversation history.\n"
        "Respond with your analysis or findings directly. If you cannot perform the task, state the reason clearly.\n"
        "Your response will be attributed to your specialized role in the conversation."
    )