# agents/tools.py
import os
from dotenv import load_dotenv
import json
from typing import Annotated, List

from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
# from langchain_google_community import GoogleSearchRun # Keep commented unless needed

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from apify_client import ApifyClient
from langchain_apify import ApifyWrapper
from langchain_core.documents import Document

load_dotenv()

APIFY_API_TOKEN=os.getenv('APIFY_API_TOKEN') # Renamed for clarity, ensure .env matches
if not APIFY_API_TOKEN:
    raise ValueError("APIFY_API_KEY hasn't been set in environment variables ")

try:
    # Ensure the env var name matches exactly what's in your .env
    linkedin_cookie_str = os.getenv('LINKEDIN_COOKIE', '{}')
    LINKEDIN_COOKIE = json.loads(linkedin_cookie_str)
except json.JSONDecodeError:
    raise ValueError(f"Invalid LINKEDIN_COOKIE format. Must be a valid JSON string. Received: {linkedin_cookie_str}")

repl = PythonREPL()
apify = ApifyWrapper(apify_api_token=APIFY_API_TOKEN)


basic_search_tool = DuckDuckGoSearchRun()
# google_serach_tool = GoogleSearchRun()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute. Can be used for complex calculations or data manipulation."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"



@tool
def scrape_linkedin_profile(profile_url: str) -> List[Document]:
    """
    Scrapes a LinkedIn profile using the 2SyF0bVxmgGr8IVCZ Actor on Apify.
    Only use this tool if you have a valid LinkedIn profile URL starting with 'https://www.linkedin.com/in/'.
    Returns the scraped profile data as a list of Documents.
    """
    print(f"--- Scraping LinkedIn Profile: {profile_url} ---")  # Debug print

    if not profile_url or not profile_url.startswith('https://www.linkedin.com/in/'):
        return [Document(page_content=f"Error: Invalid LinkedIn profile URL provided: {profile_url}. Must start with 'https://www.linkedin.com/in/'")]

    client = ApifyClient(APIFY_API_TOKEN)

    run_input = {
        "profileUrls": [profile_url]
    }

    try:
        run = client.actor("2SyF0bVxmgGr8IVCZ").call(run_input=run_input)
        dataset_items = client.dataset(run["defaultDatasetId"]).iterate_items()

        documents = []
        for item in dataset_items:
            doc = Document(
                page_content=json.dumps(item, indent=2),
                metadata={"source": "LinkedIn", "url": profile_url}
            )
            documents.append(doc)

        if not documents:
            return [Document(page_content=f"Error: No data returned from LinkedIn scraper for URL: {profile_url}. Check Apify run logs.")]

        print(f"--- Scraping Successful. Found {len(documents)} document(s). ---")
        return documents

    except Exception as e:
        print(f"--- Scraping Failed: {e} ---")
        return [Document(page_content=f"Error: Failed to scrape LinkedIn profile {profile_url}. Details: {str(e)}")]
# expose tools - ensure names match the variables
# Making a list for easier distribution to agents
all_tools = [scrape_linkedin_profile, basic_search_tool, python_repl_tool]

__all__ = [
    'python_repl_tool',
    'scrape_linkedin_profile',
    'basic_search_tool',
    'all_tools' # Export the list too
]
