from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tools import basic_search_tool
from utils import State, make_system_prompt

def create_career_guidance_agent():
    model = ChatVertexAI(model="gemini-2.0-flash-001")
    
    def career_guidance_node(state: State):
        """
        Provide career counseling and skill gap analysis
        """
        profile_data = state.get('profile_data')
        target_role = state.get('target_role')
        
        if not profile_data or not target_role:
            return {"messages": [{"role": "system", "content": "Missing profile data or target role."}]}
        
        # Search for skill requirements and learning resources
        skill_search = basic_search_tool.invoke(f"{target_role} required skills and learning paths")
        
        # Career guidance analysis
        system_prompt = make_system_prompt(
            " Provide comprehensive career counseling based on the profile and target role. "
            "Identify missing skills, suggest learning resources, and outline potential career paths."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Profile Data: {profile_data}\nTarget Role: {target_role}\nSkill Research: {skill_search}"}
        ]
        
        career_guidance = model.invoke(messages)
        
        return {
            "messages": [{"role": "assistant", "content": career_guidance.content}]
        }
    
    return career_guidance_node

# Create the agent
career_guidance_agent = create_career_guidance_agent()
