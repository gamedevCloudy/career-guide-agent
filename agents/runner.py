# agents/runner.py
import os
import functools
from dotenv import load_dotenv
from typing import List, Sequence

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .utils import AgentState, memory, SUPERVISOR_SYSTEM_PROMPT, AgentName
from .tools import all_tools # Import the list of all tools
from .profile_analysis_agent import create_profile_analysis_agent, profile_analysis_node
from .job_fit_agent import create_job_fit_agent, job_fit_node
from .career_guidance_agent import create_career_guidance_agent, career_guidance_node

load_dotenv()

# ---- Agent and Graph Definition ----

# Define agent names consistently
PROFILE_ANALYZER = "ProfileAnalyzer"
JOB_FIT_ANALYZER = "JobFitAnalyzer"
CAREER_ADVISOR = "CareerAdvisor"
SUPERVISOR = "Supervisor"

# List of worker agent members
members = [PROFILE_ANALYZER, JOB_FIT_ANALYZER, CAREER_ADVISOR]

def create_agent_supervisor(llm: ChatVertexAI, system_prompt: str, agent_names: List[str]):
    """Creates the supervisor prompt template."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"Given the conversation history, who should act next? Or should we FINISH? Select one of: {', '.join(agent_names + ['FINISH'])}")
    ])
    return prompt | llm

def create_career_optimization_graph():
    """Creates the agentic graph with a supervisor."""
    llm = ChatVertexAI(model="gemini-1.5-flash-001") # Using a capable model for supervisor

    # Create Agent Executors
    profile_agent = create_profile_analysis_agent(llm)
    job_fit_agent = create_job_fit_agent(llm)
    career_agent = create_career_guidance_agent(llm)

    # Define Agent Nodes using functools.partial to bind the agent executor
    # These functions now just need to be callable with the state
    profile_analysis_node_func = functools.partial(profile_analysis_node, agent=profile_agent, name=PROFILE_ANALYZER)
    job_fit_node_func = functools.partial(job_fit_node, agent=job_fit_agent, name=JOB_FIT_ANALYZER)
    career_guidance_node_func = functools.partial(career_guidance_node, agent=career_agent, name=CAREER_ADVISOR)

    # Create Supervisor Chain
    supervisor_chain = create_agent_supervisor(
        llm,
        SUPERVISOR_SYSTEM_PROMPT.format(members=', '.join(members)),
        members
    )

    # Define the graph
    workflow = StateGraph(AgentState)

    # Add Agent Nodes
    workflow.add_node(PROFILE_ANALYZER, profile_analysis_node_func)
    workflow.add_node(JOB_FIT_ANALYZER, job_fit_node_func)
    workflow.add_node(CAREER_ADVISOR, career_guidance_node_func)
    workflow.add_node(SUPERVISOR, supervisor_chain)

    # Define Edges - Agents route back to supervisor
    workflow.add_edge(PROFILE_ANALYZER, SUPERVISOR)
    workflow.add_edge(JOB_FIT_ANALYZER, SUPERVISOR)
    workflow.add_edge(CAREER_ADVISOR, SUPERVISOR)

    # Define Conditional Entry Point from Supervisor
    workflow.add_conditional_edges(
        SUPERVISOR,
        lambda state: state['next_agent'], # The supervisor's output decides the next node
        {
            PROFILE_ANALYZER: PROFILE_ANALYZER,
            JOB_FIT_ANALYZER: JOB_FIT_ANALYZER,
            CAREER_ADVISOR: CAREER_ADVISOR,
            "FINISH": END,
        }
    )

    # Set Supervisor as the entry point
    workflow.set_entry_point(SUPERVISOR)

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
            # Ensure output is serializable for printing if needed
            # print(output) # Can be verbose
            print("-" * 20)
            # Keep track of the latest state
            # In stream, the 'output' is the *update* to the state for that node.
            # The full state isn't directly in the event dictionary keys like this.
            # We need to reconstruct or just grab the final state after the loop.

    # After streaming, get the final state
    final_state = graph.get_state(config)
    print("--- Workflow Complete ---")

    return final_state


def main():
    # Example usage
    profile_url = "https://www.linkedin.com/in/aayush-chaudhary-2b7b99208/" # Replace with a valid test profile if needed
    target_role = "AI Research Engineer"
    thread_id = 2 # Use a different ID for a new conversation

    final_state = run_career_optimization(profile_url, target_role, thread_id=thread_id)

    # Print out the final messages
    print("\n--- Final Conversation History ---")
    if final_state and final_state.get('messages'):
        for message in final_state['messages']:
             # Check message type for better formatting (optional)
            role = "Unknown"
            content = ""
            if hasattr(message, 'role'): # Anthropic/OpenAI format
                role = message.role
            elif hasattr(message, 'type'): # LangChain message types
                 role = message.type
            if hasattr(message, 'content'):
                 content = message.content

            # Handle potential complex content (like tool calls/results if not filtered)
            if isinstance(content, list): # e.g., tool calls
                content_str = "\n".join([str(c) for c in content])
            else:
                content_str = str(content)

            print(f"**{role.upper()}**: {content_str}\n")

    else:
        print("No final state or messages found.")

if __name__ == "__main__":
    main()