import json 
import requests
import streamlit as st
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query to perform")
    
class SearchTool(BaseTool):
    name: str = "Search The internet"
    description: str = """Useful to search the internet for information."""
    args_schema: type[BaseModel] = SearchQuery
    
    def _run(self, query: str) -> str:
        try:
            load_dotenv()
            
            top_results_to_return = 5
            search_url = "https://google.serper.dev/search"
            payload = json.dumps({
                "q": query,
            })
            headers = {
                'content-Type': 'application/json',
                'X-API-KEY': os.getenv("SERPER_API_KEY")
            }
            response = requests.request("POST", search_url, headers=headers, data=payload)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('organic', [])
                if not results:
                    return "No results found."
                
                # Extract the top results
                top_results = []
                for result in results[:top_results_to_return]:
                    title = result.get('title')
                    link = result.get('link')
                    snippet = result.get('snippet')
                    top_results.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n")
                
                return "\n".join(top_results) if top_results else "No valid results found"
        except Exception as e:
            return f"Error: {e}"
        
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("SearchTool does not support async run.")