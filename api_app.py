from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from crewai import Agent, Crew, Task, LLM
import os 
from dotenv import load_dotenv
from functools import lru_cache

from trip_agents import TripAgents
from trip_tasks import TripTasks

# Load the environment variables from the .env file
load_dotenv()

app = FastAPI(
    title="VacAIgent API",
    description="An API for planning trips using AI agents.",
    version="1.0.0",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for trip planning
class TripRequest(BaseModel):
    origin: str = Field(..., example="Bangalore, India", description="Your current location or origin city.")
    destination: str = Field(..., example="Bali, Indonesia", description="Destination city and country.")
    start_date: str = Field(..., example="2025-01-10", description="Start date of your trip in YYYY-MM-DD format.")
    end_date: str = Field(..., example="2025-01-16", description="End date of your trip in YYYY-MM-DD format. If not provided, a 6-day trip is assumed.")
    interests: str = Field(..., example="2 adults who love swimming, dancing, hiking", description="Your interests and trip details.")
 
# Response model for trip planning   
class TripResponse(BaseModel):
    status: str
    message: str
    itinerary: Optional[str] = None
    error: Optional[str] = None
    

class Settings:
    def __init__(self):
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        self.BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
@lru_cache()
def get_settings():
    return Settings()

def validate_api_keys(settings: Settings = Depends(get_settings)):
    required_keys = {
        'GEMINI_API_KEY': settings.SERPER_API_KEY,
        'SERPER_API_KEY': settings.SERPER_API_KEY,
        'BROWSERLESS_API_KEY': settings.BROWSERLESS_API_KEY,
        'GROQ_API_KEY': settings.GROQ_API_KEY,
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    if missing_keys:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required environment variables: {', '.join(missing_keys)}. Please set them in your .env file or environment."
        )
    return settings

class TripCrew:
    def __init__(self, origin: str, destination: str, date_range: str, interests: str):
        self.origin = origin
        self.destination = destination
        self.date_range = date_range
        self.interests = interests
        self.llm = LLM(model="")
        
    def run(self):
        try: 
            agents = TripAgents(llm = self.llm)
            tasks = TripTasks()
            
            city_selection_agent = agents.city_selection_agent()
            local_expert_agent = agents.local_expert()
            travel_concierge_agent = agents.travel_coincerge()
            
            identify_task = tasks.identify_task(
                city_selection_agent,
                self.origin,
                self.destination,
                self.interests,
                self.date_range
            )
            
            gather_task = tasks.gather_task(
                local_expert_agent,
                self.origin,
                self.interests,
                self.date_range
            )
            
            plan_task = tasks.plan_task(
                travel_concierge_agent,
                self.origin,
                self.interests,
                self.date_range
            )
            # Initialize the crew with agents and tasks
            crew = Crew(
                agents=[city_selection_agent, local_expert_agent, travel_concierge_agent],
                tasks=[identify_task, gather_task, plan_task],
                verbose=True
            )
            
            # Kick off the crew to execute the tasks
            result = crew.kickoff()
            return result.raw if hasattr(result, 'raw') else str(result)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error initializing agents: {str(e)}"
            )
            
@app.post("/api/v1/plan_trip", response_model=TripResponse)
async def plan_trip(
        trip_request: TripRequest, 
        settings: Settings = Depends(validate_api_keys)
    ):
        # validate dates
        if trip_request.end_date <= trip_request.start_date:
            raise HTTPException(
                status_code=400,
                detail="End date must be after start date."
            )
        # format date range 
        date_range = f"{trip_request.start_date} to {trip_request.end_date}"
        
        try:
            trip_crew = TripCrew(
                origin=trip_request.origin,
                destination=trip_request.destination,
                date_range=date_range,
                interests=trip_request.interests
            )
            
            itinerary = trip_crew.run()
            return TripResponse(
                status="success",
                message="Trip planning completed successfully.",
                itinerary=itinerary
            )
        except HTTPException as http_exc:
            return TripResponse(
                status="error",
                message="An error occurred during trip planning.",
                error=str(http_exc.detail)
            )
            
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "VacAIgent API is running smoothly."
    }
    
@app.get("/")
async def root():
    return {
        "message": "Welcome to VacAIgent API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }
    
# Swagger 
@app.get("/")
async def root():
    return {
        "message": "Welcome to VacAIgent API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }
   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)