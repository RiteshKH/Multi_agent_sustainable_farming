from pydantic_ai import Agent, RunContext
from typing import List, Dict, Optional
from dataclasses import dataclass
import logfire
import json
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model

logfire.configure(send_to_logfire='if-token-present')

model = get_model()

@dataclass
class MarketDeps:
    market_data: pd.DataFrame

system_prompt = """
You are a market intelligence specialist who helps farmers make informed decisions.

Use the tools provided to analyze market trends, forecast seasonal changes, and identify opportunities
for sustainable and profitable farming.

Always explain the reasoning behind your insights and recommendations.

Format your response in a clear, organized way with actionable insights and data references.

Never ask for clarification on any piece of information before providing insights, just make
your best guess for any parameters that you aren't sure of.
"""

market_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=MarketDeps,
    retries=2
)

@market_agent.tool
async def track_crop_pricing(ctx: RunContext[MarketDeps]) -> str:
    """Analyze crop pricing and demand patterns."""
    market_data = ctx.deps.market_data
    pricing_trends = market_data.groupby('Product').agg({
        'Market_Price_per_ton': 'mean',
        'Demand_Index': 'mean',
        'Supply_Index': 'mean'
    }).reset_index().to_dict(orient='records')
    return json.dumps({"pricing_trends": pricing_trends})

@market_agent.tool
async def forecast_seasonal_changes(ctx: RunContext[MarketDeps]) -> str:
    """Forecast seasonal market changes."""
    market_data = ctx.deps.market_data
    seasonal_forecast = market_data.groupby('Seasonal_Factor').agg({
        'Market_Price_per_ton': 'mean',
        'Demand_Index': 'mean',
        'Weather_Impact_Score': 'mean'
    }).reset_index().to_dict(orient='records')
    return json.dumps({"seasonal_forecast": seasonal_forecast})

@market_agent.tool
async def identify_high_value_crops(ctx: RunContext[MarketDeps]) -> str:
    """Identify high-value crops with lower environmental impact."""
    market_data = ctx.deps.market_data
    high_value_crops = market_data.loc[
        market_data['Economic_Indicator'] > 1
    ].groupby('Product').agg({
        'Market_Price_per_ton': 'mean',
        'Weather_Impact_Score': 'mean'
    }).reset_index().to_dict(orient='records')
    return json.dumps({"high_value_crops": high_value_crops})

@market_agent.tool
async def calculate_profitability_ratios(ctx: RunContext[MarketDeps]) -> str:
    """Calculate profitability-to-sustainability ratios for different crops."""
    market_data = ctx.deps.market_data
    market_data['Profitability_to_Sustainability'] = (
        market_data['Market_Price_per_ton'] / market_data['Weather_Impact_Score']
    )
    profitability_ratios = market_data.groupby('Product').agg({
        'Profitability_to_Sustainability': 'mean'
    }).reset_index().to_dict(orient='records')
    return json.dumps({"profitability_ratios": profitability_ratios})

# def load_data() -> MarketDeps:
#     market_data = pd.read_csv('mrd_tiny.csv')  # Updated to use mrd_tiny.csv
#     return MarketDeps(market_data=market_data)

# def get_market_intelligence():
#     data = load_data()
#     insights = market_agent.run(data)
#     return insights
