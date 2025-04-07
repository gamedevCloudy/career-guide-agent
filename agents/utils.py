import os 
import sqlite3
from dotenv import load_dotenv
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()

DATABASE_URI=os.getenv('DATABASE_URI')

class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    profile_url: str
    profile_data: Document
    target_role: str

def create_sqlite_memory():
    try:
        conn = sqlite3.connect(DATABASE_URI.replace('sqlite:///', ''))
        return SqliteSaver(conn=conn)
    except Exception as e:
        print(f"Error creating SQLite memory: {e}")
        return None

memory = create_sqlite_memory()

def make_system_prompt(suffix: str) -> str:
    base_prompt=  (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
    )

    return f"{base_prompt}{suffix}".strip()