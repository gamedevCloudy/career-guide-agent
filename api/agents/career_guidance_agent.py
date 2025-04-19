# agents/career_guidance_agent.py
from langchain_google_vertexai import ChatVertexAI

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent

from agents.tools import basic_search_tool # Only needs search
from agents.utils import make_agent_system_prompt

from typing import Literal
from langgraph.types import Command

# Tools specific to this agent
guidance_tools = [basic_search_tool]

llm = ChatVertexAI(model_name="gemini-2.0-flash")



system_prompt= make_agent_system_prompt("""
    Your only role is to give career guidance based on Linkedin Profile and target role
                                        
    you can use basic_search_tool to get information about the target profile and give specific guidance. 
    Always provide incremental guide and actionable steps forward.                                    
    """)
career_guidance_agent  = create_react_agent(llm, tools=guidance_tools, prompt=system_prompt)

def career_guidance_node(state: MessagesState) -> Command[Literal['__end__']]: 
    result = career_guidance_agent.invoke(state)
    print('=' * 50 )
    print('Executed: Career Guidance'  )
    print(result)
    print('=' * 50 )
    return Command(
        update={
            "messages": [
                HumanMessage(content=result['messages'][-1].content, name="career_guide")
            ]
        },
        goto="__end__"
    )


__all__ = [
    'career_guidance_node'
]