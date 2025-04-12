# agents/job_fit_agent.py
from langchain_google_vertexai import ChatVertexAI

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import  HumanMessage
from langgraph.graph import MessagesState
from typing import List, Optional, Literal
from langgraph.types import Command

from agents.tools import basic_search_tool # Only needs search
from agents.utils import make_agent_system_prompt

# Tools specific to this agent
job_fit_tools = [basic_search_tool]




llm = ChatVertexAI(model_name='gemini-2.0-flash-001')

# system_prompt = make_agent_system_prompt(
#        "Analyze the fit between the user's profile data (available in the conversation history, likely provided by the ProfileAnalyzer) and a specified target job role. "
#        "Use the search tool to find standard job descriptions, required skills, and industry expectations for the target role. "
#        "Compare the profile data against these standards. "
#        "Provide a detailed analysis including: "
#        "1. A qualitative assessment of the fit (e.g., Strong Match, Good Match, Needs Improvement). "
#        "2. Identification of key skill gaps. "
#        "3. Suggestions for specific improvements to the profile or skills needed to bridge the gap. "
#        "4. Consider the user's current experience level when providing feedback and suggesting career steps. "
#        "If profile data is missing or insufficient in the conversation history, state that you cannot perform the analysis."
#     )

system_prompt= make_agent_system_prompt("""
    Your only role is to give a job fit based on Target role and linkedin profile
    
    You can you basic_search_tool to look up information from the internet regarding the target role                                    
    """)

job_fit_agent = create_react_agent(llm, tools=job_fit_tools, prompt=system_prompt)




def job_fit_node(state: MessagesState) -> Command[Literal['__end__']]: 
    result = job_fit_agent.invoke(state)
    print('=' * 50 )
    print('Executed: Job Fit'  )
    print(result)
    print('=' * 50 )
    return Command(
        update={
            "messages": [
                HumanMessage(content=result['messages'][-1].content, name="job_fit")
            ]
        },
        goto="__end__"
    )

    

__all__ = [
    'job_fit_node'
]