# agents/runner.py
import os
import functools
from dotenv import load_dotenv
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import  HumanMessage
from langchain_google_vertexai import ChatVertexAI



from utils import AgentState, memory, SUPERVISOR_SYSTEM_PROMPT, AgentName
from tools import all_tools # Import the list of all tools
from profile_analysis_agent import profile_analysis_node
from job_fit_agent import job_fit_node
from career_guidance_agent import career_guidance_node

from typing import Literal
from langgraph.types import Command

from langchain_core.language_models.chat_models import BaseChatModel


load_dotenv()

# ---- Agent and Graph Definition ----

# Define agent names consistently
PROFILE_ANALYZER = "ProfileAnalyzer"
JOB_FIT_ANALYZER = "JobFitAnalyzer"
CAREER_ADVISOR = "CareerAdvisor"
SUPERVISOR = "Supervisor"

# List of worker agent members
members = [PROFILE_ANALYZER, JOB_FIT_ANALYZER, CAREER_ADVISOR]



def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}.",
        "Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node

def create_career_optimization_graph():
    """Creates the agentic graph with a supervisor."""
    llm = ChatVertexAI(model="gemini-2.0-flash-001") # Using a capable model for supervisor

    supervisor_node = make_supervisor_node(llm=llm, members=members)
    
    # Define the graph
    workflow = StateGraph(AgentState)

    # Add Agent Nodes
    workflow.add_node(PROFILE_ANALYZER, profile_analysis_node)
    workflow.add_node(JOB_FIT_ANALYZER, job_fit_node)
    workflow.add_node(CAREER_ADVISOR, career_guidance_node)
    workflow.add_node(SUPERVISOR, supervisor_node)

    # Define Edges - Agents route back to supervisor
    workflow.add_edge(START, SUPERVISOR)

    # Compile the graph
    graph = workflow.compile(checkpointer=memory)

    return graph

def run_career_optimization(profile_url: str, target_role: str, thread_id: int = 1):
    """
    Run the career optimization agentic workflow.

    Args:
        profile_url (str): LinkedIn profile URL.
        target_role (str): Target job role.
        thread_id (int): Conversation thread ID for persistence.

    Returns:
        dict: Final state of the workflow.
    """
    graph = create_career_optimization_graph()

    # Initial user message
    initial_message = HumanMessage(
        content=f"Please analyze my LinkedIn profile: {profile_url} and assess my fit for the target role: '{target_role}'. Provide career guidance based on this."
    )

    # Initial state - supervisor will read this first
    initial_state = AgentState(
        messages=[initial_message],
        apify_api_token=os.environ['APIFY_API_KEY'],
        profile_data=None,
        next_agent=None # Supervisor decides first step
    )

    config = {'configurable': {'thread_id': str(thread_id)}} # Thread ID must be string

    print(f"\n--- Starting Career Optimization for Thread {thread_id} ---")
    print(f"Profile URL: {profile_url}")
    print(f"Target Role: {target_role}")
    print("-" * 20)

    # Stream or invoke the graph
    final_state = None
    # Using stream to see the flow
    for event in graph.stream(initial_state, config=config):
        for node, output in event.items():
            print(f"--- Output from Node: {node} ---")
            
            print("-" * 20)
            

    # After streaming, get the final state
    final_state = graph.get_state(config)
    print("--- Workflow Complete ---")

    return final_state


def main():
    # Example usage
    profile_url = "https://www.linkedin.com/in/aayush-chaudhary-2b7b99208/" # Replace with a valid test profile if needed
    target_role = "AI Research Engineer"
    thread_id = 1 # Use a different ID for a new conversation

    final_state = run_career_optimization(profile_url, target_role, thread_id=thread_id)

    print(final_state)

if __name__ == "__main__":
    main()