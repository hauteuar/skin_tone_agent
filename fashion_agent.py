from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("API_KEY")


agent = Agent(tools=[DuckDuckGo()], show_tool_calls=True, markdown=True)
agent.print_response("Whats happening in Fashion trends?", stream=True)