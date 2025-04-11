# agents/profile_analysis_agent.py
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from typing import List, Optional, Literal
from langgraph.types import Command


from agents.tools import scrape_linkedin_profile, basic_search_tool
from agents.utils import make_agent_system_prompt



# Tools specific to this agent
profile_tools = [scrape_linkedin_profile, basic_search_tool]

system_prompt = make_agent_system_prompt(
        "Your only Job is to provide a analysis of LinkedIn profile for ranking in SEO as well as optimizing for HR's preferences."
        "Analyze a given LinkedIn profile URL. "
        "First, use the 'scrape_linkedin_profile' tool to fetch the profile data. "
        "It has been pre-configured with required API keys, can be called directly"
        "Use Search tool to look up things associated with profile - like best practices."
        "If scraping is successful, analyze the content for strengths, weaknesses, gaps, and inconsistencies across all sections (Summary, Experience, Education, Skills, etc.). "
        "If scraping fails or no URL is provided in the history, state that you cannot proceed without valid profile data. "
    )

llm = ChatVertexAI(model="gemini-2.0-flash")

profile_analysis_agent = create_react_agent(llm, tools=profile_tools, prompt=system_prompt)

def profile_analysis_node(state: MessagesState) -> Command[Literal['supervisor']]: 
    result = profile_analysis_agent.invoke(state)
    print('=' * 50 )
    print('Executed: Profile analyser'  )
    print(result)
    print('=' * 50 )
    return Command(
        update={
            "messages": [
                HumanMessage(content=result['messages'][-1].content, name="profile_analysis")
            ]
        },
        goto="supervisor"
    )

__all__ = [
    'profile_analysis_node'
]