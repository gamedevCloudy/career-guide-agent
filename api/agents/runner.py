# agents/runner.py
import os
import functools
from dotenv import load_dotenv
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import  SystemMessage, HumanMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI



from agents.utils import AgentState, memory, SUPERVISOR_SYSTEM_PROMPT, AgentName
from agents.tools import all_tools # Import the list of all tools
from agents.profile_analysis_agent import profile_analysis_node
from agents.job_fit_agent import job_fit_node
from agents.career_guidance_agent import career_guidance_node

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

MAX_CONVERSATION_TURNS=5

# List of worker agent members
members = [PROFILE_ANALYZER, JOB_FIT_ANALYZER, CAREER_ADVISOR, COUNCELLER]


def make_conversation_node(llm:BaseChatModel, members: list[str]) -> str: 
    options = ['FINISH'] + members

    system_prompt = """
    <role>Counseller</role>
    <name>Ria</name>

    <goal>help our clients and users optimize thier profile and provide career guidance</goal>

    <team>
    {members}
    </team>

    <task>
    <first_message>
    - greet the Client 
    - ask their name
    - ask them their target role
    - ask the client for LinkedIn Profile URL 
    </first_message>
    - give them basics details what you can do with your team. 
    - communicate with {members} to get detials from the team regarding our client 
    - you can get their profile details, profile analysis job fit and career guidance tips 
    </task>

    <tone>
    - friendly and human like 
    - should be hopeful and try to help client most 
    </tone>

    <tips>
    - go in depth
    - do not be generic, be specific and provide correct guidance. 
    </tips>
    """

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def counceller_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""

        messages = state["messages"]
        
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})
    
    return counceller_node

# agents/runner.py (inside make_supervisor_node)

def make_supervisor_node(llm: BaseChatModel, members: list[str], counceller_name: str) -> callable:

    options = members + [counceller_name]

    system_prompt = (
        f"You are a supervisor managing a conversation between the following workers: {members}. "
        f"The overall interaction is managed by the {counceller_name}."
        "Given the conversation history, decide which worker should act next to fulfill the user's request. "
        "Your options are: " + ", ".join(options) + "."
        f"If the user's request requires analysis from a worker (e.g., {', '.join(members)}), route to that worker. "
        f"If all necessary analysis from the workers is complete based on the conversation history, route back to the {counceller_name} to synthesize the information and respond to the user."
        "Only choose one worker or the counceller per turn."

    )


    class Router(TypedDict):
        """Route to the next worker or back to the Counceller."""
        next: Literal[*options] 

    def supervisor_node(state: AgentState) -> Command[Literal[*options]]:
        print(f"\n--- Executing Supervisor ---")
        print(f"Input State Keys: {state.keys()}")
        last_message = state["messages"][-1]
        print(f"Input Last Message ({type(last_message).__name__}): {last_message.content}")

        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"] # Pass the actual message objects

        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        # if goto == "FINISH":
        #     goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node

def create_career_optimization_graph():
    """Creates the agentic graph with a supervisor and counceller."""
    llm = ChatVertexAI(model="gemini-2.0-flash-001")

    # Create nodes
    counceller_node = make_conversation_node(llm=llm, members=[SUPERVISOR])
    supervisor_node = make_supervisor_node(llm=llm, members=members, counceller_name=COUNCELLER)
    
    # Define the graph
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node(COUNCELLER, counceller_node)
    workflow.add_node(SUPERVISOR, supervisor_node)
    workflow.add_node(PROFILE_ANALYZER, profile_analysis_node)
    workflow.add_node(JOB_FIT_ANALYZER, job_fit_node)
    workflow.add_node(CAREER_ADVISOR, career_guidance_node)


    # Start with Counceller
    workflow.add_edge(START, COUNCELLER)


    # Agents must route back to Supervisor
    workflow.add_edge(PROFILE_ANALYZER, SUPERVISOR)
    workflow.add_edge(JOB_FIT_ANALYZER, SUPERVISOR)
    workflow.add_edge(CAREER_ADVISOR, SUPERVISOR)

    # Supervisor must route back to Counceller
    workflow.add_edge(SUPERVISOR, "Counceller")

    # Compile the graph
    graph = workflow.compile(checkpointer=memory)
    
    try: 
        png_image = graph.get_graph().draw_mermaid_png()
        with open("career_optimization_graph.png", "wb") as f:
            f.write(png_image)
    except Exception as e: 
        print(str(e))
    
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
            
            print("-" * 20)
            

    # After streaming, get the final state
    final_state = graph.get_state(config)
    print("--- Workflow Complete ---")

    return final_state


def run_career_conversation_step(user_input: str, thread_id: str):
    graph = create_career_optimization_graph()

    # Pull conversation state from memory if it exists
    config = {'configurable': {'thread_id': thread_id}}
    
    try:
        # Attempt to get existing state
        current_state = graph.get_state(config)
        
        # Add new user message to existing messages
        current_state.values['messages'].append(HumanMessage(content=user_input))
    except Exception:
        # If no existing state, create initial state with system and user message
        current_state = AgentState(
            messages=[
                HumanMessage(content=user_input)
            ],
            profile_data=None,
            next_agent=None
        )

    # Stream the graph and process the entire conversation
    final_state = None
    for event in graph.stream(current_state, config=config):
        final_state = event

    # Retrieve the last AI message
    for msg in reversed(graph.get_state(config).values['messages']):
        if isinstance(msg, AIMessage):
            return msg.content

    return "I'm ready to help you with your career goals. Could you provide more details?"



def main():
    # Example usage
    profile_url = "https://www.linkedin.com/in/aayush-chaudhary-2b7b99208/" # Replace with a valid test profile if needed
    target_role = "AI Research Engineer"
    thread_id = 1 # Use a different ID for a new conversation

    final_state = run_career_optimization(profile_url, target_role, thread_id=thread_id)

    print(final_state)

if __name__ == "__main__":
    main()