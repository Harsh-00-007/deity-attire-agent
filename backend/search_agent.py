import os
import requests
from typing import List, Optional
from pydantic import BaseModel, Field

# LangChain imports for modern LLM orchestration
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY, SERPAPI_API_KEY)
load_dotenv()

class SearchQuery(BaseModel):
    """
    Pydantic schema to enforce the LLM outputs exactly what we need.
    This guarantees we don't get conversational junk back from the model,
    just the highly optimized search string.
    """
    optimized_query: str = Field(
        description="The highly optimized search query for Google Images or Pinterest to find the specific design."
    )
    justification: str = Field(
        description="A brief, 1-sentence reason why these keywords were chosen."
    )

class TrendResearcherAgent:
    """
    An autonomous agent that acts as a Design Researcher.
    It takes vague client requests, optimizes them using an LLM, 
    and fetches relevant high-quality image URLs from the web.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Initializes the agent with a specific LLM model.
        """
        # We use a smaller, faster model for reasoning to save latency and costs.
        try:
            self.llm = ChatGroq(model=model_name, temperature=0.2)
        except Exception as e:
            print(f"[!] Warning: LLM initialization failed. Is OPENAI_API_KEY set? Error: {e}")
            self.llm = None
            
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")

    def _refine_user_prompt(self, user_input: str) -> SearchQuery:
        """
        Phase 1: Reasoning. 
        Uses the LLM to translate a normal user request into an SEO-optimized image search query.
        """
        if not self.llm:
            raise ValueError("LLM is not initialized. Cannot refine prompt.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Fashion and Design Researcher specializing in Indian cultural attire for all Hindu deities (Radha Krishna, Ram Darbar, Laddu Gopal)"
                       "Your job is to take a client's vague request and turn it into a highly specific, keyword-dense search query optimized for finding high-resolution reference images. "
                       "Include terms like 'high resolution', 'design', 'zardosi', 'embroidery', or 'pinterest' if applicable to the context."),
            ("human", "Client request: {user_input}")
        ])

        # Modern LangChain: Bind the Pydantic schema to the model to guarantee structured JSON output
        structured_llm = self.llm.with_structured_output(SearchQuery)
        chain = prompt | structured_llm
        
        print("[*] Agent is analyzing the request and reasoning...")
        return chain.invoke({"user_input": user_input})

    def _execute_image_search(self, optimized_query: str, max_results: int = 5) -> List[str]:
        """
        Phase 2: Execution.
        Calls an external Search API (like SerpApi's Google Images engine) to fetch image URLs.
        """
        print(f"[*] Agent executing search for: '{optimized_query}'")
        
        # FAANG Best Practice: Graceful degradation. If the API key is missing, 
        # return mock data so the app doesn't crash during local testing.
        if not self.serpapi_key:
            print("[!] SERPAPI_API_KEY not found in environment. Returning mock fallback images.")
            return [
                "https://placehold.co/400x400/ea580c/ffffff?text=Mock+Poshak+1",
                "https://placehold.co/400x500/ca8a04/ffffff?text=Mock+Zardosi+2",
                "https://placehold.co/400x300/16a34a/ffffff?text=Mock+Design+3"
            ]

        # Actual implementation for SerpApi (Google Images)
        search_url = "https://serpapi.com/search"
        params = {
            "engine": "google_images",
            "q": optimized_query,
            "api_key": self.serpapi_key,
            "num": max_results
        }

        try:
            response = requests.get(search_url, params=params, timeout=58)
            response.raise_for_status()
            data = response.json()
            
            # Extract the actual image URLs from the JSON payload
            image_results = data.get("images_results", [])
            urls = []
            
            for img in image_results:
                url = img.get("original")
                # Filter out strict domains that block Python scripts
                if url and "lookaside.instagram.com" not in url and "fbsbx.com" not in url:
                    urls.append(url)
                
                if len(urls) >= max_results:
                    break
                    
            return urls
            
        except requests.exceptions.RequestException as e:
            print(f"[!] Error connecting to Search API: {e}")
            return []

    def run(self, user_input: str) -> List[str]:
        """
        The main orchestration pipeline.
        """
        try:
            # Step 1: Think and refine
            structured_plan = self._refine_user_prompt(user_input)
            print(f"[+] Agent decided to search: {structured_plan.optimized_query}")
            print(f"[+] Agent reasoning: {structured_plan.justification}")
            
            # Step 2: Act and fetch
            image_urls = self._execute_image_search(structured_plan.optimized_query)
            print(f"[+] Successfully retrieved {len(image_urls)} reference images.")
            
            return image_urls
            
        except Exception as e:
            print(f"[-] Agent Workflow Failed: {e}")
            return []

# --- Quick Testing Block ---
if __name__ == "__main__":
    # This block allows you to run `python agent_search.py` in your terminal to test it
    # without needing to spin up the entire UI.
    
    agent = TrendResearcherAgent()
    
    test_query = "Find me some cool blue dresses for laddu gopal for janmashtami"
    
    print("\n--- Starting Agentic Search Run ---")
    results = agent.run(test_query)
    
    print("\n--- Final Output URLs ---")
    for idx, url in enumerate(results, 1):
        print(f"{idx}. {url}")