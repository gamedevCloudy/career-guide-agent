# agents/job_fit_agent.py
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from .tools import basic_search_tool # Only needs search
from .utils import make_agent_system_prompt, AgentState

# Tools specific to this agent
job_fit_tools = [basic_search_tool]

def create_job_fit_agent(llm: ChatVertexAI):
    """Creates the Job Fit Analysis Agent Executor."""
    system_prompt = make_agent_system_prompt(
       "Analyze the fit between the user's profile data (available in the conversation history, likely provided by the ProfileAnalyzer) and a specified target job role. "
       "Use the search tool to find standard job descriptions, required skills, and industry expectations for the target role. "
       "Compare the profile data against these standards. "
       "Provide a detailed analysis including: "
       "1. A qualitative assessment of the fit (e.g., Strong Match, Good Match, Needs Improvement). "
       "2. Identification of key skill gaps. "
       "3. Suggestions for specific improvements to the profile or skills needed to bridge the gap. "
       "4. Consider the user's current experience level when providing feedback and suggesting career steps. "
       "If profile data is missing or insufficient in the conversation history, state that you cannot perform the analysis."
    )

    llm_with_system_prompt = llm.bind(system_message=SystemMessage(content=system_prompt))
    
    agent_executor = create_react_agent(llm_with_system_prompt, job_fit_tools)
    return agent_executor

def job_fit_node(state: AgentState, agent: callable, name: str):
    """Node function to execute the job fit agent."""
    print(f"--- Executing {name} ---")
    # Check if profile data exists from previous step before invoking
    # Though the agent prompt handles this, adding a check here might be redundant but safer
    # if not state.get('profile_data'):
    #     # This shouldn't happen if the supervisor routes correctly, but as a fallback
    #     return {"messages": [("system", f"Error in {name}: Profile data not found in state.")]}

    result = agent.invoke(state)
    return {"messages": result["messages"]}