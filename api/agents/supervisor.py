# supervisor.py
from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langchain_google_vertexai import ChatVertexAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import List
from langgraph.graph import MessagesState
from agents.profile_analysis_agent import profile_analysis_node
from agents.job_fit_agent import job_fit_node
from agents.career_guidance_agent import career_guidance_node

# Use the same LLM as your agents for consistency
llm = ChatVertexAI(model="gemini-2.0-flash-001")

members = ["job_fit", "career_guide", "profile_analysis"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
"""
<task>
- Manage a conversation between the following workers: {members}
- Decide which worker should act next based on the user request
- Each worker performs a task and returns results and status
- After each worker responds, reply with FINISH
</task>

<process>
- First, execute `profile_analyser`
- If any member lacks sufficient information, reply with FINISH
- If any member needs additional information, reply with FINISH
</process>

<RULE>
- Execute only one worker at a time
- After the worker responds, always reply with FINISH
- Do not run the same worker more than once
- If the analysis is incomplete, reply with FINISH
</RULE>
"""
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]


def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    print("="*50)
    print(response)
    print("="*50)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

def create_career_graph():
    builder = StateGraph(MessagesState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)

    builder.add_node("job_fit", job_fit_node)
    builder.add_node("career_guide", career_guidance_node)
    builder.add_node("profile_analysis", profile_analysis_node)

    # Compile the graph
    return builder.compile()