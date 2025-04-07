import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_vertexai import ChatVertexAI

from utils import State, memory
from profile_analysis_agent import profile_analysis_agent
from job_fit_agent import job_fit_agent
from career_guidance_agent import career_guidance_agent

load_dotenv()

def create_career_optimization_workflow():
    """
    Create a workflow for LinkedIn profile optimization and career guidance
    """
    workflow = StateGraph(State)
    # Add nodes
    workflow.add_node("profile_analysis", profile_analysis_agent)
    workflow.add_node("job_fit_analysis", job_fit_agent)
    workflow.add_node("career_guidance", career_guidance_agent)

    # Define edges
    workflow.set_entry_point("profile_analysis")
    workflow.add_edge("profile_analysis", "job_fit_analysis")
    workflow.add_edge("job_fit_analysis", "career_guidance")
    workflow.add_edge("career_guidance", END)

    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    return app

def run_career_optimization(profile_url: str, target_role: str):
    """
    Run the career optimization workflow
    
    Args:
        profile_url (str): LinkedIn profile URL
        target_role (str): Target job role
    
    Returns:
        dict: Workflow results
    """
    workflow = create_career_optimization_workflow()
    
    # Initial input
    initial_state = {
        "messages": [],
        "profile_url": profile_url,
        "target_role": target_role,
        "name": "Career Optimization Assistant"
    }

    initial_state = State(**initial_state)
    
    # Run the workflow
    results = workflow.invoke(initial_state, config={'configurable': {'thread_id': 1}})
    
    return results

def main():
    # Example usage
    profile_url = "https://www.linkedin.com/in/aayush-chaudhary-2b7b99208/"
    target_role = "AI Reasearch Engineer"
    

    results = run_career_optimization(profile_url, target_role)
    
    # Print out the results
    for message in results.get('messages', []):
        # print(f"{message.get('role', 'Unknown')}: {message.get('content', 'No content')}")
        print(message)

if __name__ == "__main__":
    main()

