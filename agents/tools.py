import os 
from dotenv import load_dotenv
import json 

from typing import Annotated


from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_google_community import GoogleSearchRun

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from langchain_apify import ApifyWrapper
from langchain_core.documents import Document

load_dotenv()

APIFY_TOKEN=os.getenv('APIFY_API_TOKEN')
if not APIFY_TOKEN: 
    raise ValueError("Apify API key hasn't been set in environment variables ")

try:
    LINKEDIN_COOKIE = json.loads(os.getenv('LINKEDIN_COOKIE', '{}'))
except json.JSONDecodeError:
    raise ValueError("Invalid LINKEDIN_COOKIE format. Must be a valid JSON string.")

repl = PythonREPL()
apify = ApifyWrapper(apify_api_token=APIFY_TOKEN)


basic_search_tool = DuckDuckGoSearchRun()
# google_serach_tool = GoogleSearchRun()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


@tool 
def scrape_linkedin_profile(profile_url: str) -> dict:
    """
    Scrapes a LinkedIn profile using the curious_coder/linkedin-profile-scraper Actor on Apify.

    Args:
        profile_url (str): The URL of the LinkedIn profile to scrape.

    Returns:
        dict: The scraped profile data.
    """
    # check if link is valid 

    if not profile_url or not profile_url.startswith('https://www.linkedin.com/in/'):
        raise ValueError("Invalid LinkedIn profile URL. Must start with 'https://www.linkedin.com/in/'")
    
    # Prepare the input for the Actor

    actor_input = {
        "urls": [profile_url],
        "cookie": LINKEDIN_COOKIE,
        "scrapeCompany": False,
        "minDelay": 15,
        "maxDelay": 60,
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyCountry": "US"
        }
    }

    

    def map_dataset_item(item):
        return Document(page_content=json.dumps(item), metadata={"source": "LinkedIn"})

    # Call actor and get mapped dataset output
    data_loader = apify.call_actor(
        actor_id="curious_coder/linkedin-profile-scraper",
        run_input=actor_input,
        dataset_mapping_function=map_dataset_item
    )

    result = data_loader.load()

    return result




# expose tools
__all__ = [
    'python_repl_tool', 
    'scrape_linkedin_profile', 
    'basic_search_tool'
]

