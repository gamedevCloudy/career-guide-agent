# agents/runner.py
import os
import functools
from dotenv import load_dotenv
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI

from api.agents.utils import AgentState, memory, SUPERVISOR_SYSTEM_PROMPT, AgentName
from api.agents.tools import all_tools
from api.agents.profile_analysis_agent import profile_analysis_node
from api.agents.job_fit_agent import job_fit_node
from api.agents.career_guidance_agent import career_guidance_node

from typing import Literal
from langgraph.types import Command

from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

# ---- Agent and Graph Definition ----

# Define agent names consistently
PROFILE_ANALYZER = "ProfileAnalyzer"
JOB_FIT_ANALYZER = "JobFitAnalyzer"
CAREER_ADVISOR = "CareerAdvisor"
COUNCELLER = "Counceller"
SUPERVISOR = "Supervisor"

MAX_CONVERSATION_TURNS = 5

# List of worker agent members
WORKER_MEMBERS = [PROFILE_ANALYZER, JOB_FIT_ANALYZER, CAREER_ADVISOR]
ALL_MEMBERS = WORKER_MEMBERS + [COUNCELLER]

def make_conversation_node(llm: BaseChatModel) -> callable:
    system_prompt = """
    <role>Counseller</role>
    <name>Ria</name>

    <goal>help our clients and users optimize their profile and provide career guidance</goal>

    <task>
    <first_message>
    - greet the Client 
    - ask their name
    - ask them their target role
    - ask the client for LinkedIn Profile URL 
    </first_message>
    - give them basics details what you can do with your team. 
    - you can get their profile details, profile analysis job fit and career guidance tips 
    </task>

    <tone>
    - friendly and human like 
    - should be hopeful and try to help client most 
    </tone>
    <check>
        Check state, once you have the analysis details repsonsd back, "I am ready to guide you now. Please ask"
    </check>
    <tips>
    - go in depth
    - do not be generic, be specific and provide correct guidance. 
    </tips>
    """

    def counceller_node(state: AgentState):
        # Get the last message in the conversation
        messages = state["messages"]
        
        # Prepare the system prompt and messages for the LLM
        formatted_messages = [
            {"role": "system", "content": system_prompt},
        ] + messages
        
        # Generate response from the LLM
        response = llm.invoke(formatted_messages)
        
        # Return updated state with the new message
        return {
            "messages": state["messages"] + [
                AIMessage(content=response.content, name="Ria")
            ]
        }
    
    return counceller_node

def make_supervisor_node(llm: BaseChatModel, worker_members: list[str]) -> callable:
    options = worker_members + [COUNCELLER]

    system_prompt = (
        f"You are a supervisor managing a conversation between a counselor and specialized workers.\n\n"
        f"IMPORTANT: DO NOT route to the same worker multiple times in a row unless the user explicitly asks for it.\n\n"
        f"Available workers: {', '.join(worker_members)}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. If profile analysis hasn't been done and user needs LinkedIn profile analysis, route to {PROFILE_ANALYZER}\n"
        f"2. If profile analysis is done but job fit analysis hasn't been done, route to {JOB_FIT_ANALYZER}\n"
        f"3. If both profile and job fit analyses are done but career guidance hasn't been provided, route to {CAREER_ADVISOR}\n"
        f"4. If all analyses are complete or for general conversation, route to {COUNCELLER}\n\n"
    )

    class Router(TypedDict):
        """Route to the next worker or back to the Counceller."""
        next: Literal[tuple(options)]

    def supervisor_node(state: AgentState) -> AgentState:
        print(f"\n--- Executing Supervisor ---")
        
        # Prepare messages for the LLM
        formatted_messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add actual conversation messages
        for msg in state["messages"]:
            msg_content = {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
            if hasattr(msg, "name") and msg.name:
                msg_content["name"] = msg.name
            formatted_messages.append(msg_content)

        # Get routing decision
        response = llm.with_structured_output(Router).invoke(formatted_messages)
        next_agent = response["next"]
        
        print(f"Supervisor decided to route to: {next_agent}")
        
        return {"next": next_agent}

    return supervisor_node

def create_career_optimization_graph():
    """Creates the agentic graph with a supervisor and counceller."""
    llm = ChatVertexAI(model="gemini-2.0-flash-001")

    # Create nodes
    counceller_node = make_conversation_node(llm=llm)
    supervisor_node = make_supervisor_node(llm=llm, worker_members=WORKER_MEMBERS)
    
    # Define the graph
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node(COUNCELLER, counceller_node)
    workflow.add_node(SUPERVISOR, supervisor_node)
    workflow.add_node(PROFILE_ANALYZER, profile_analysis_node)
    workflow.add_node(JOB_FIT_ANALYZER, job_fit_node)
    workflow.add_node(CAREER_ADVISOR, career_guidance_node)

    # Add conditional routing
    # Every message starts at the supervisor to determine routing
    workflow.add_edge(START, SUPERVISOR)
    
    # Define conditional edges from the supervisor
    def route_from_supervisor(state):
        next_agent = state.get("next")
        
        # Add safety check to prevent loops
        last_agents = [msg.name for msg in state["messages"][-3:] if hasattr(msg, "name")]
        if next_agent in last_agents and last_agents.count(next_agent) >= 2:
            # If same agent appears multiple times recently, route to Counceller instead
            print(f"Loop detected with {next_agent}, routing to Counceller instead")
            return SUPERVISOR
            
        return next_agent
    # Supervisor routes to appropriate agent based on decision
    workflow.add_conditional_edges(
        SUPERVISOR,
        route_from_supervisor,
        {
            PROFILE_ANALYZER: PROFILE_ANALYZER,
            JOB_FIT_ANALYZER: JOB_FIT_ANALYZER,
            CAREER_ADVISOR: CAREER_ADVISOR,
            COUNCELLER: COUNCELLER
        }
    )
    
    # All agents report back to the supervisor for next decision
    workflow.add_edge(PROFILE_ANALYZER, SUPERVISOR)
    workflow.add_edge(JOB_FIT_ANALYZER, SUPERVISOR)
    workflow.add_edge(CAREER_ADVISOR, SUPERVISOR)
    
    # Counceller is the final step in the chain, goes back to START (supervisor) for the next turn
    workflow.add_edge(COUNCELLER, END)
    
    # Compile the graph
    graph = workflow.compile(checkpointer=memory)
    
    try: 
        png_image = graph.get_graph().draw_mermaid_png()
        with open("assets/career_optimization_graph.png", "wb") as f:
            f.write(png_image)
    except Exception as e: 
        print(f"Error generating graph image: {str(e)}")
    
    return graph

def run_career_optimization(profile_url: str, target_role: str, thread_id: str = "1"):
    """
    Run the career optimization agentic workflow.

    Args:
        profile_url (str): LinkedIn profile URL.
        target_role (str): Target job role.
        thread_id (str): Conversation thread ID for persistence.

    Returns:
        dict: Final state of the workflow.
    """
    graph = create_career_optimization_graph()

    # Initial user message
    initial_message = HumanMessage(
        content=f"Please analyze my LinkedIn profile: {profile_url} and assess my fit for the target role: '{target_role}'. Provide career guidance based on this."
    )

    # Initial state
    initial_state = AgentState(
        messages=[initial_message],
        profile_data=None,
        next=SUPERVISOR  # Start with supervisor to route
    )

    config = {'configurable': {'thread_id': thread_id}}

    print(f"\n--- Starting Career Optimization for Thread {thread_id} ---")
    print(f"Profile URL: {profile_url}")
    print(f"Target Role: {target_role}")
    print("-" * 20)

    # Using stream to see the flow
    for event in graph.stream(initial_state, config=config):
        for node, output in event.items():
            print(f"--- Output from Node: {node} ---")
            if output and "messages" in output and output["messages"]:
                last_message = output["messages"][-1]
                print(f"Message: {last_message.content[:100]}...")
            print("-" * 20)

    # Get the final state
    final_state = graph.get_state(config)
    print("--- Workflow Complete ---")

    return final_state

def chat_with_agent(message: str, thread_id: str = "1"):
    """
    Function to handle single chat messages for the FastAPI endpoint.
    
    Args:
        message (str): The user's message
        thread_id (str): The conversation thread ID
        
    Returns:
        dict: Updated state with the agent's response
    """
    graph = create_career_optimization_graph()
    config = {'configurable': {'thread_id': thread_id}}
    
    try:
        # Try to get existing state
        current_state = graph.get_state(config)
        
        # Add the new message to existing conversation
        human_message = HumanMessage(content=message)
        current_state["messages"].append(human_message)
        
    except Exception as e:
        print(f"Starting new conversation: {str(e)}")
        # Start a new conversation
        human_message = HumanMessage(content=message)
        current_state = AgentState(
            messages=[human_message],
            profile_data=None,
            next=SUPERVISOR
        )
    
    # Process through the graph
    final_state = graph.invoke(current_state, config=config)
    
    return final_state

def main():
    # Example usage
    profile_url = "https://www.linkedin.com/in/aayush-chaudhary-2b7b99208/"
    target_role = "AI Research Engineer"
    thread_id = "1"

    final_state = run_career_optimization(profile_url, target_role, thread_id=thread_id)
    print(final_state)

if __name__ == "__main__":
    main()