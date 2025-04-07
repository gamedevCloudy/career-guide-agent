# agents/utils.py
import os
import sqlite3
from dotenv import load_dotenv
from typing import Annotated, List, Optional, Literal

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver # Using MemorySaver for simplicity now

load_dotenv()

DATABASE_URI=os.getenv('DATABASE_URI', 'agent_checkpoint.sqlite') # Provide default

# Define Agent names for routing
AgentName = Literal["ProfileAnalyzer", "JobFitAnalyzer", "CareerAdvisor", "FINISH"]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # profile_url: Optional[str] # Can be derived from initial message
    # target_role: Optional[str] # Can be derived from initial message
    profile_data: Optional[List[Document]] # Store scraped data here
    # Add a field for the router to decide the next step
    next_agent: Optional[AgentName]

def create_sqlite_memory():
    try:
        # Ensure the directory exists if DATABASE_URI includes a path
        db_dir = os.path.dirname(DATABASE_URI)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        conn = sqlite3.connect(DATABASE_URI, check_same_thread=False) # check_same_thread=False is often needed
        return SqliteSaver(conn=conn)
    except Exception as e:
        print(f"Error creating SQLite memory at {DATABASE_URI}: {e}")
        print("Falling back to MemorySaver.")
        return MemorySaver()

memory = MemorySaver() # Using MemorySaver for easier testing, switch back if persistence is needed
# memory = create_sqlite_memory()


# This base prompt is more suitable for the supervisor
SUPERVISOR_SYSTEM_PROMPT = (
    "You are a supervisor tasked with managing a conversation between the following specialized agents: {members}. "
    "Given the user request, determine which agent should act next or if the task is complete. "
    "Each agent will perform its task based on the conversation history and available tools. "
    "When the goal is achieved and no further action is needed, respond with 'FINISH'."
    "Do not delegate the same task over and over again. If the agent is unable to fulfill the request, and you have tried all options, respond with 'FINISH'."
    "Only respond with the name of the agent to delegate to or 'FINISH'."
)

# Individual agents might need more specific prompts defined in their respective files.
# This helper can still be used for consistency if desired.
def make_agent_system_prompt(role_description: str) -> str:
    # Simplified prompt for REACT agents
    return (
        f"You are a helpful AI assistant. Your specific role is: {role_description}\n"
        "Use the provided tools to perform your task based on the current conversation history. "
        "Respond with your analysis or findings. If you cannot perform the task, state the reason clearly. "
        "Do not delegate work to other agents; the supervisor handles delegation."
        # "If you have completed your specific part of the task, make sure your response reflects this."
        # React agent framework handles the thought/action/observation loop,
        # so explicit instruction about FINAL ANSWER might not be needed here.
    )