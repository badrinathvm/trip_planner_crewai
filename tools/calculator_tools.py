from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class CalculationInput(BaseModel):
    operation: str = Field(..., description="The mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "Make Calculations"
    description: str = """Useful to perform any mathematical calculations, 
    like sum, minus, multiplication, division, etc.
    The input should be a mathematical expression, e.g. '200*7' or '5000/2*10'"""
    args_schema: type[BaseModel] = CalculationInput
    
    def _run(self, operation: str) -> str:
        return eval(operation)
    
    def _arun(self, operation: str) -> str:
        raise NotImplementedError("CalculatorTool does not support async run.")