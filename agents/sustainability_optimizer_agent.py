from typing import List, Dict, Optional, Any, Callable
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pydantic_ai import Agent, RunContext
import os
import sys
import logfire

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model

logfire.configure(send_to_logfire='if-token-present')

model = get_model()

# Define the data models
class FarmData(BaseModel):
    """Farm agricultural data model"""
    Farm_ID: int
    Soil_pH: float
    Soil_Moisture: float
    Temperature_C: float
    Rainfall_mm: float
    Crop_Type: str
    Fertilizer_Usage_kg: float
    Pesticide_Usage_kg: float
    Crop_Yield_ton: float
    Sustainability_Score: float

class SustainabilityRecommendation(BaseModel):
    """Recommendations for improving sustainability"""
    Farm_ID: int
    Crop_Type: str
    current_fertilizer: float
    current_pesticide: float
    optimal_fertilizer_usage: float = Field(..., description="Recommended fertilizer usage in kg")
    optimal_pesticide_usage: float = Field(..., description="Recommended pesticide usage in kg")
    estimated_sustainability_improvement: float = Field(..., description="Estimated improvement in sustainability score")
    estimated_yield_impact: float = Field(..., description="Estimated impact on crop yield in percentage")
    rationale: str = Field(..., description="Explanation for the recommendations")
    soil_recommendations: Dict[str, Any] = Field(..., description="Recommendations for soil management")

class SoilRecommendation(BaseModel):
    """Soil management recommendations"""
    summary: str
    ph_adjustment: str
    moisture_adjustment: str
    optimal_ph_range: List[float]
    optimal_moisture_range: List[float]

class CorrelationAnalysis(BaseModel):
    """Analysis of correlations between farming practices and sustainability"""
    fertilizer_correlation: float
    pesticide_correlation: float
    soil_ph_correlation: float
    soil_moisture_correlation: float
    insights: str

class CropSustainabilityInsight(BaseModel):
    """Sustainability insights for a specific crop type"""
    crop_type: str
    avg_sustainability_score: float
    avg_fertilizer_usage: float
    avg_pesticide_usage: float
    avg_yield: float
    insights: str

# Define the dependencies that will be made available to the LLM
class FarmingDeps(BaseModel):
    farm_data: List[Dict[str, Any]] = []
    load_farm_data: Optional[Callable] = None
    analyze_correlations: Optional[Callable] = None
    get_crop_sustainability_insights: Optional[Callable] = None
    get_soil_recommendations: Optional[Callable] = None
    generate_optimization_recommendation: Optional[Callable] = None

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like pandas.DataFrame


# class SustainabilityOptimizerAgent(BaseModel):
#     """Agent that analyzes farm data to optimize sustainability using an LLM"""
    
#     def run(self, user_prompt: str) -> str:
#         """Main entry point for the agent"""

# Define system prompt for the LLM
system_prompt = """
You are an AI Sustainability Optimizer for agriculture. Your goal is to analyze farm data and provide 
recommendations to improve sustainability scores while maintaining crop yields.

You have access to the following tools:
1. load_farm_data: Load data from a CSV file
2. analyze_correlations: Analyze correlations between farming practices and sustainability
3. get_crop_sustainability_insights: Get insights for specific crop types
4. get_soil_recommendations: Get recommendations for soil management
5. generate_optimization_recommendation: Generate optimization recommendations for a farm

Your task is to:
1. Load and analyze the farm data
2. Identify patterns between farming practices and sustainability scores
3. Generate recommendations to improve sustainability for specific farms
4. Provide clear, actionable advice with rationales

Important considerations:
- Sustainability improvements should not significantly reduce crop yields (aim for less than 5% yield reduction)
- Focus on optimizing fertilizer and pesticide usage
- Consider soil conditions (pH and moisture) for each crop type
- Provide recommendations that are practical and implementable
"""

# Initialize the LLM-based agent
sustainability_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=FarmingDeps,
    retries=2
)


@sustainability_agent.tool
def load_farm_data(ctx: RunContext[FarmingDeps]) -> List[Dict[str, Any]]:
    """
    Load and parse farm data from CSV file
    
    Args:
        file_path: Path to the CSV file containing farm data
        
    Returns:
        List of farm data dictionaries
    """
    farm_data = ctx.deps.farm_data
    return farm_data

@sustainability_agent.tool
def analyze_correlations(ctx: RunContext[FarmingDeps]) -> CorrelationAnalysis:
    """
    Analyze correlations between farming practices and sustainability scores
    
    Args:
        farm_data: List of farm data dictionaries
        
    Returns:
        Analysis of correlations with insights
    """
    df = pd.DataFrame(ctx.deps.farm_data)
    
    # Calculate correlations
    fertilizer_corr = df['Fertilizer_Usage_kg'].corr(df['Sustainability_Score'])
    pesticide_corr = df['Pesticide_Usage_kg'].corr(df['Sustainability_Score'])
    ph_corr = df['Soil_pH'].corr(df['Sustainability_Score'])
    moisture_corr = df['Soil_Moisture'].corr(df['Sustainability_Score'])
    
    # Generate insights based on correlations
    insights = []
    
    if fertilizer_corr < -0.3:
        insights.append("Lower fertilizer usage correlates with higher sustainability scores.")
    elif fertilizer_corr > 0.3:
        insights.append("Higher fertilizer usage correlates with higher sustainability scores, suggesting efficient usage.")
        
    if pesticide_corr < -0.3:
        insights.append("Lower pesticide usage correlates with higher sustainability scores.")
    elif pesticide_corr > 0.3:
        insights.append("Higher pesticide usage correlates with higher sustainability scores, suggesting targeted application.")
        
    if abs(ph_corr) > 0.3:
        if ph_corr > 0:
            insights.append("Higher soil pH correlates with higher sustainability scores.")
        else:
            insights.append("Lower soil pH correlates with higher sustainability scores.")
            
    if abs(moisture_corr) > 0.3:
        if moisture_corr > 0:
            insights.append("Higher soil moisture correlates with higher sustainability scores.")
        else:
            insights.append("Lower soil moisture correlates with higher sustainability scores.")
    
    return CorrelationAnalysis(
        fertilizer_correlation=float(fertilizer_corr),
        pesticide_correlation=float(pesticide_corr),
        soil_ph_correlation=float(ph_corr),
        soil_moisture_correlation=float(moisture_corr),
        insights=". ".join(insights)
    )

@sustainability_agent.tool
def get_crop_sustainability_insights(ctx: RunContext[FarmingDeps], crop_type: str) -> CropSustainabilityInsight:
    """
    Get sustainability insights for a specific crop type
    
    Args:
        farm_data: List of farm data dictionaries
        crop_type: The crop type to analyze
        
    Returns:
        Sustainability insights for the specified crop type
    """
    df = pd.DataFrame(ctx.deps.farm_data)
    
    # Filter for the specific crop
    crop_df = df[df['Crop_Type'] == crop_type]
    
    if crop_df.empty:
        return CropSustainabilityInsight(
            crop_type=crop_type,
            avg_sustainability_score=0,
            avg_fertilizer_usage=0,
            avg_pesticide_usage=0,
            avg_yield=0,
            insights=f"No data found for crop type: {crop_type}"
        )
    
    # Calculate averages
    avg_sustainability = crop_df['Sustainability_Score'].mean()
    avg_fertilizer = crop_df['Fertilizer_Usage_kg'].mean()
    avg_pesticide = crop_df['Pesticide_Usage_kg'].mean()
    avg_yield = crop_df['Crop_Yield_ton'].mean()
    
    # Find farms with high sustainability scores for this crop
    high_sustainability = crop_df[crop_df['Sustainability_Score'] > avg_sustainability]
    
    # Generate insights
    insights = []
    insights.append(f"Average sustainability score for {crop_type} is {avg_sustainability:.2f}.")
    
    if not high_sustainability.empty:
        avg_high_fert = high_sustainability['Fertilizer_Usage_kg'].mean()
        avg_high_pest = high_sustainability['Pesticide_Usage_kg'].mean()
        
        fert_diff = avg_high_fert - avg_fertilizer
        pest_diff = avg_high_pest - avg_pesticide
        
        if abs(fert_diff) > 10:
            direction = "more" if fert_diff > 0 else "less"
            insights.append(f"Farms with higher sustainability scores use {direction} fertilizer than average.")
            
        if abs(pest_diff) > 1:
            direction = "more" if pest_diff > 0 else "less"
            insights.append(f"Farms with higher sustainability scores use {direction} pesticide than average.")
    
    return CropSustainabilityInsight(
        crop_type=crop_type,
        avg_sustainability_score=float(avg_sustainability),
        avg_fertilizer_usage=float(avg_fertilizer),
        avg_pesticide_usage=float(avg_pesticide),
        avg_yield=float(avg_yield),
        insights=". ".join(insights)
    )

@sustainability_agent.tool
def get_soil_recommendations(ctx: RunContext[FarmingDeps], crop_type: str, current_ph: float, current_moisture: float) -> SoilRecommendation:
    """
    Get soil management recommendations for specific crop types
    
    Args:
        crop_type: Type of crop (Wheat, Corn, Soybean, Rice)
        current_ph: Current soil pH level
        current_moisture: Current soil moisture percentage
        
    Returns:
        Soil management recommendations
    """
    optimal_conditions = {
        "Wheat": {"ph_min": 6.0, "ph_max": 7.0, "moisture_min": 20, "moisture_max": 40},
        "Corn": {"ph_min": 5.8, "ph_max": 7.0, "moisture_min": 25, "moisture_max": 45},
        "Soybean": {"ph_min": 6.0, "ph_max": 7.0, "moisture_min": 20, "moisture_max": 50},
        "Rice": {"ph_min": 5.5, "ph_max": 6.5, "moisture_min": 30, "moisture_max": 60}
    }
    
    if crop_type not in optimal_conditions:
        return SoilRecommendation(
            summary="No specific soil recommendations available for this crop type.",
            ph_adjustment="No data",
            moisture_adjustment="No data",
            optimal_ph_range=[0, 0],
            optimal_moisture_range=[0, 0]
        )
    
    optimal = optimal_conditions[crop_type]
    
    # Calculate pH adjustment
    if current_ph < optimal["ph_min"]:
        ph_adjustment = f"Increase soil pH from {current_ph:.1f} to {optimal['ph_min']:.1f}-{optimal['ph_max']:.1f} range by adding lime"
    elif current_ph > optimal["ph_max"]:
        ph_adjustment = f"Decrease soil pH from {current_ph:.1f} to {optimal['ph_min']:.1f}-{optimal['ph_max']:.1f} range by adding sulfur or organic matter"
    else:
        ph_adjustment = f"Maintain current soil pH of {current_ph:.1f}, which is optimal for {crop_type}"
    
    # Calculate moisture adjustment
    if current_moisture < optimal["moisture_min"]:
        moisture_adjustment = f"Increase soil moisture from {current_moisture:.1f}% to {optimal['moisture_min']:.1f}-{optimal['moisture_max']:.1f}% range through improved irrigation"
    elif current_moisture > optimal["moisture_max"]:
        moisture_adjustment = f"Decrease soil moisture from {current_moisture:.1f}% to {optimal['moisture_min']:.1f}-{optimal['moisture_max']:.1f}% range through improved drainage"
    else:
        moisture_adjustment = f"Maintain current soil moisture of {current_moisture:.1f}%, which is optimal for {crop_type}"
    
    summary = f"Soil adjustments for {crop_type}: {ph_adjustment}. {moisture_adjustment}."
    
    return SoilRecommendation(
        summary=summary,
        ph_adjustment=ph_adjustment,
        moisture_adjustment=moisture_adjustment,
        optimal_ph_range=[optimal["ph_min"], optimal["ph_max"]],
        optimal_moisture_range=[optimal["moisture_min"], optimal["moisture_max"]]
    )

@sustainability_agent.tool
def generate_optimization_recommendation(
    ctx: RunContext[FarmingDeps],
    crop_insights: CropSustainabilityInsight,
    reduction_target: float = 0.2
) -> SustainabilityRecommendation:
    """
    Generate recommendations for optimal resource usage based on farm data and crop insights
    
    Args:
        farm_data: Dictionary containing data for a specific farm
        crop_insights: Sustainability insights for the farm's crop type
        reduction_target: Target reduction percentage for inputs (default: 0.2 or 20%)
        
    Returns:
        Sustainability recommendations for the farm
    """
    farm_data = pd.DataFrame(ctx.deps.farm_data)
    # Extract farm data
    farm_id = farm_data["Farm_ID"]
    crop_type = farm_data["Crop_Type"]
    current_fertilizer = farm_data["Fertilizer_Usage_kg"]
    current_pesticide = farm_data["Pesticide_Usage_kg"]
    current_sustainability = farm_data["Sustainability_Score"]
    current_yield = farm_data["Crop_Yield_ton"]
    
    # Calculate optimal values based on high-performing farms for this crop
    # (In a real implementation, this would be more sophisticated)
    optimal_fertilizer = current_fertilizer * (1 - reduction_target)
    optimal_pesticide = current_pesticide * (1 - reduction_target)
    
    # Conservative estimate of sustainability improvement
    est_sustainability_improvement = reduction_target * 10
    
    # Conservative estimate of yield impact
    est_yield_impact = -reduction_target * 10 / 2  # Assume yield reduction is half the input reduction
    
    # Get soil recommendations
    soil_recs = get_soil_recommendations(
        crop_type=crop_type, 
        current_ph=farm_data["Soil_pH"], 
        current_moisture=farm_data["Soil_Moisture"]
    )
    
    # Generate rationale
    rationale = (
        f"Based on analysis of similar farms growing {crop_type}, reducing fertilizer from "
        f"{current_fertilizer:.2f}kg to {optimal_fertilizer:.2f}kg and pesticides from "
        f"{current_pesticide:.2f}kg to {optimal_pesticide:.2f}kg can improve sustainability score by "
        f"{est_sustainability_improvement:.2f} points with a projected {est_yield_impact:.2f}% impact on yield. "
        f"This recommendation takes into account that the average sustainability score for {crop_type} is "
        f"{crop_insights.avg_sustainability_score:.2f} and your current score is {current_sustainability:.2f}."
    )
    
    return SustainabilityRecommendation(
        Farm_ID=farm_id,
        Crop_Type=crop_type,
        current_fertilizer=current_fertilizer,
        current_pesticide=current_pesticide,
        optimal_fertilizer_usage=optimal_fertilizer,
        optimal_pesticide_usage=optimal_pesticide,
        estimated_sustainability_improvement=est_sustainability_improvement,
        estimated_yield_impact=est_yield_impact,
        rationale=rationale,
        soil_recommendations=soil_recs.dict()
    )


# Run the agent
# result = sustainability_agent(user_prompt)

# return result

# # Example usage
# if __name__ == "__main__":
#     import sys
#     import logfire

#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from utils import get_model

#     logfire.configure(send_to_logfire='if-token-present')

#     model = get_model()
#     # Create the agent
#     agent = SustainabilityOptimizerAgent()
    
#     # Example context
#     context = {
#         "farm_data_path": "fad_tiny.csv",
#         "farm_id": 1,  # Optional: specify a farm to analyze
#         "reduction_target": 0.15  # Target 15% reduction in inputs
#     }
    
#     # Run the agent
#     results = agent.run(context)
    
#     # Print results
#     print(results)