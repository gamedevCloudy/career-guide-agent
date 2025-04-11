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

system_prompt = make_agent_system_prompt(
        "Provide comprehensive career guidance based on the user's profile analysis and job fit assessment (available in the conversation history). "
        "Use the search tool to find information on career paths, required skills for advancement, and potential learning resources (courses, certifications, communities) relevant to the target role and identified skill gaps. "
        "Synthesize the information from the history and your research to: "
        "1. Outline potential career trajectories from the user's current state towards the target role and beyond. "
        "2. Suggest concrete next steps, including skill development and networking strategies. "
        "3. Recommend specific learning resources. "
        "If prior analysis (profile, job fit) is missing or insufficient in the conversation history, state that you cannot provide full guidance."
)

career_guidance_agent  = create_react_agent(llm, tools=guidance_tools, prompt=system_prompt)

def career_guidance_node(state: MessagesState) -> Command[Literal['supervisor']]: 
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
        goto="supervisor"
    )


__all__ = [
    'career_guidance_node'
]