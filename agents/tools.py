import os 
from dotenv import load_dotenv
import json 

from typing import Annotated


from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from langchain_apify import ApifyWrapper
from langchain_core.documents import Document

load_dotenv()

APIFY_TOKEN=os.environ['APIFY_API_TOKEN']
LINKEDIN_COOKIE=json.loads(os.environ['LINKEDIN_COOKIE'])
repl = PythonREPL()
apify = ApifyWrapper(apify_api_token=APIFY_TOKEN)
search_tool = DuckDuckGoSearchRun()

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





# Using the Profile data loading tool 
# profile_url = "https://www.linkedin.com/in/aayush-chaudhary-2b7b99208/"
# profile_data = scrape_linkedin_profile.invoke(profile_url)


info = search_tool.invoke("who is DemonKingSwarn")
print(info)