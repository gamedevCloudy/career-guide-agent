# agents/profile_analysis_agent.py
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from .tools import scrape_linkedin_profile, basic_search_tool
from .utils import make_agent_system_prompt, AgentState

# Tools specific to this agent
profile_tools = [scrape_linkedin_profile, basic_search_tool]

def create_profile_analysis_agent(llm: ChatVertexAI):
    """Creates the Profile Analysis Agent Executor."""
    system_prompt = make_agent_system_prompt(
        "Analyze a given LinkedIn profile URL. "
        "First, use the 'scrape_linkedin_profile' tool to fetch the profile data. "
        "If scraping is successful, analyze the content for strengths, weaknesses, gaps, and inconsistencies across all sections (Summary, Experience, Education, Skills, etc.). "
        "If scraping fails or no URL is provided in the history, state that you cannot proceed without valid profile data. "
        "You can use the search tool to look up general best practices for LinkedIn profiles if needed for comparison."
    )

    agent_executor = create_react_agent(llm, profile_tools, messages_modifier=system_prompt)
    return agent_executor

def profile_analysis_node(state: AgentState, agent: callable, name: str):
    """Node function to execute the profile analysis agent."""
    print(f"--- Executing {name} ---")
    # The agent requires the messages list.
    result = agent.invoke(state)

    # The react agent invocation returns the final response message in the 'messages' list
    # We need to update the overall state with this new message.
    # The scraped data should ideally be extracted and placed in state['profile_data'] if successful.
    # Let's try to parse the agent's output or check tool calls for scraped data.

    # Simplistic approach: Assume last message contains analysis or error.
    # A more robust approach would involve parsing tool outputs if the agent framework allows access.
    # For now, let the supervisor handle checking if scraping happened based on messages.

    return {"messages": result["messages"]}