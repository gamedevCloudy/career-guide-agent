# agents/profile_analysis_agent.py
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

from typing import List, Optional, Literal
from langgraph.types import Command


from agents.tools import scrape_linkedin_profile, basic_search_tool
from agents.utils import make_agent_system_prompt, AgentState

# Tools specific to this agent
profile_tools = [scrape_linkedin_profile, basic_search_tool]

system_prompt = make_agent_system_prompt(
        "Analyze a given LinkedIn profile URL. "
        "First, use the 'scrape_linkedin_profile' tool to fetch the profile data. "
        "It has been pre-configured with required API keys, can be called directly"
        "Use Search tool to look up things associated with profile - like best practices."
        "If scraping is successful, analyze the content for strengths, weaknesses, gaps, and inconsistencies across all sections (Summary, Experience, Education, Skills, etc.). "
        "If scraping fails or no URL is provided in the history, state that you cannot proceed without valid profile data. "
        "You can use the search tool to look up general best practices for LinkedIn profiles if needed for comparison."
    )

llm = ChatVertexAI(model="gemini-2.0-flash")

profile_analysis_agent = create_react_agent(llm, tools=profile_tools, prompt=system_prompt)


def profile_analysis_node(state: AgentState, agent: callable, name: str):
    """Node function to execute the profile analysis agent."""
    print(f"--- Executing {name} ---")
    # The agent requires the messages list.
    result = agent.invoke(state)

    return {"messages": result["messages"]}

def profile_analysis_node(state: AgentState) -> Command[Literal['Supervisor']]: 
    result = profile_analysis_agent.invoke(state)
    
    print(result)
    return Command(
          update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="CareerAdvisor")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="Supervisor",
    )

__all__ = [
    'profile_analysis_node'
]