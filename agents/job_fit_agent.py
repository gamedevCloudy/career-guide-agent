from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool 
from langgraph.prebuilt import create_react_agent
from tools import basic_search_tool
from utils import State, make_system_prompt

def create_job_fit_agent(): 
    model = ChatVertexAI(model_name="gemini-2.0-flash-001")

    def job_fit_node(state: State): 
        """
        Analzye job fit and generate improvement suggesitions 
        """

        target_role = state.get('target_role')
        profile_data = state.get('profile_data')

        if not profile_data or not target_role:
            return {"messages": [{"role": "system", "content": "Missing profile data or target role."}]}
                    
        job_description_search = basic_search_tool.invoke(f"{target_role} job description industry standards")

        system_prompt = make_system_prompt(
            """<goal>Compare and contrast the LinkedIn profile of a with the target job role</goal>
            <tasks>
            - generate a match score out of 10 
            - identify skill gaps the user and suggest specific improvements
            - response should be in depth focusing on various aspects of the career
            - take into account the current experience level when analysing 
            - include a potential career trajectory with your reposne 
            eg. target: Tech Lead 
            and if user is still a student, then suggest every step in middle ie SDE-1 to SDE 5, then gaining expertice in technical project management, going niche at current role, this will also require no job hopping etc.  
            </tasks>

            Note: be detailed

            output: 

            class JobFit: 
                feedback: str 
            
            return:  JobFit
            """
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Profile Data: {profile_data}\nTarget Role: {target_role}\nJob Description Search: {job_description_search}"}
        ]
        
        job_fit_analysis = model.invoke(messages)
        
        return {
            "messages": [{"role": "assistant", "content": job_fit_analysis.content}]
        }
    
    return job_fit_node


job_fit_agent = create_job_fit_agent()

        