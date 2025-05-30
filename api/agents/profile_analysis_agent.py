# agents/profile_analysis_agent.py
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, END
from typing import List, Optional, Literal
from langgraph.types import Command


from agents.tools import scrape_linkedin_profile, basic_search_tool
from agents.utils import make_agent_system_prompt



# Tools specific to this agent
profile_tools = [scrape_linkedin_profile, basic_search_tool]



system_prompt= make_agent_system_prompt("Your only role is to scrape user's profile and give a SEO and HR optimised profile review You can you basic_search_tool to look up information from the internet regarding the target role")

llm = ChatVertexAI(model="gemini-2.0-flash")

profile_analysis_agent = create_react_agent(llm, tools=profile_tools, prompt=system_prompt)

def profile_analysis_node(state: MessagesState) -> Command[Literal['__end__']]: 
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
        goto='__end__'
    )

__all__ = [
    'profile_analysis_node'
]