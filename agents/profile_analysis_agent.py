from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool 
from langgraph.prebuilt import create_react_agent
from agents.utils import scrape_linkedin_profile, google_search_tool
from agents.utils import State, make_system_prompt

from pydantic import BaseModel, Field


class SectionAnalysis(BaseModel): 
    section: str = Field(description="name of the profile section")
    analysis: str = Field(description="a clear, descriptive and actionable analysis of the profile")

def create_profile_analysis_agent(): 
    
    model = ChatVertexAI(model_name="gemini-2.0-flash-001")

    def profile_analysis_node(state: State): 
        """
        Analyze LinkedIn profile for gaps and inconsistances 
        """
        profile_url = state.get('profile_url')

        if not profile_url: 
            return {"messeges": [{"role": "system", "content": "No LinkedIn profile URL provided"}]}
        
        profile_data = scrape_linkedin_profile(profile_url)

        system_prompt = make_system_prompt(
            """
            Perform a comprehensive analysis of LinkedIn profile
            Identify strengths, weaknesses, gaps and inconsistancies in all sections. 
            
            class SectionAnalysis(BaseModel): 
                section: str = Field(description="name of the profile section")
                analysis: str = Field(description="a clear, descriptive and actionable analysis of the profile")

            return: list[SectionAnalysis]
            """
        )

        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": f"Analyze this LinkedIn Profile: {profile_data}"}
        ]

        analysis = model.invoke(messages)

        return {
            "messages": [{"role": "assistant", 
                          "content": analysis.content
                          }]
        }

    return profile_analysis_node

profile_analysis_agent = create_profile_analysis_agent()