from crewai import Agent, Task, Crew
from langchain.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain_community.llms import Ollama

# Initialize the tools and models
search_tool = DuckDuckGoSearchRun()
llm = Ollama(model="mistral")  # You can also use "llama2" if you prefer

# Create a research agent
researcher = Agent(
    role='Research Analyst',
    goal='Search and gather accurate information about fashion trends',
    backstory="""You are an expert fashion research analyst with a talent for finding 
    accurate and relevant information about current trends. You excel at crafting effective 
    search queries and summarizing fashion findings.""",
    tools=[
        Tool(
            name='web_search',
            func=search_tool.run,
            description='Search the web for fashion information'
        )
    ],
    llm=llm,
    verbose=True
)

# Create a writer agent to summarize findings
writer = Agent(
    role='Fashion Writer',
    goal='Create clear and concise summaries of fashion trends and advice',
    backstory="""You are a skilled fashion writer who excels at taking complex 
    trend information and making it easy to understand. You create well-structured 
    summaries that highlight the most important fashion points.""",
    llm=llm,
    verbose=True
)

def run_search_crew(query):
    # Create tasks for the agents
    research_task = Task(
        description=f"""Search for information about: {query}
        - Use the web_search tool to find relevant fashion information
        - Focus on reliable fashion sources and recent trends
        - Collect key fashion details and style tips
        - Note any emerging trends
        """,
        agent=researcher,
        expected_output="""A detailed report containing:
        - Current fashion trends
        - Key style elements
        - Popular colors and patterns
        - Styling recommendations
        """
    )

    writing_task = Task(
        description=f"""Create a comprehensive fashion summary about: {query}
        - Organize trends logically
        - Highlight key fashion points
        - Include practical styling tips
        - Address different body types and occasions
        """,
        agent=writer,
        expected_output="""A well-structured fashion guide including:
        - Trend analysis
        - Styling suggestions
        - Practical tips
        - Accessory recommendations
        """
    )

    # Create and run the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True
    )
    
    result = crew.kickoff()
    return result

# Example usage
if __name__ == "__main__":
    # First install required packages:
    # pip install crewai langchain langchain-community duckduckgo-search
    # And install Ollama from: https://ollama.ai/download
    
    search_query = "Latest fashion trends for 2024"
    results = run_search_crew(search_query)
    print(results)