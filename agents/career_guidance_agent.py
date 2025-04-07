# agents/career_guidance_agent.py
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from .tools import basic_search_tool # Only needs search
from .utils import make_agent_system_prompt, AgentState

# Tools specific to this agent
guidance_tools = [basic_search_tool]

def create_career_guidance_agent(llm: ChatVertexAI):
    """Creates the Career Guidance Agent Executor."""
    system_prompt = make_agent_system_prompt(
        "Provide comprehensive career guidance based on the user's profile analysis and job fit assessment (available in the conversation history). "
        "Use the search tool to find information on career paths, required skills for advancement, and potential learning resources (courses, certifications, communities) relevant to the target role and identified skill gaps. "
        "Synthesize the information from the history and your research to: "
        "1. Outline potential career trajectories from the user's current state towards the target role and beyond. "
        "2. Suggest concrete next steps, including skill development and networking strategies. "
        "3. Recommend specific learning resources. "
        "If prior analysis (profile, job fit) is missing or insufficient in the conversation history, state that you cannot provide full guidance."
    )
    agent_executor = create_react_agent(llm, guidance_tools, messages_modifier=system_prompt)
    return agent_executor

def career_guidance_node(state: AgentState, agent: callable, name: str):
    """Node function to execute the career guidance agent."""
    print(f"--- Executing {name} ---")
    result = agent.invoke(state)
    return {"messages": result["messages"]}