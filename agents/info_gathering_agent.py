from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import Any, List, Dict
from dataclasses import dataclass
import logfire
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model

logfire.configure(send_to_logfire='if-token-present')

model = get_model()

class FarmDetails(BaseModel):
    """Details"""
    response: str = Field(description='The response to give back to the user if they did not give all the necessary details')
    # all_details_given: bool = Field(description='True if the user has given all the necessary details, otherwise false')

system_prompt = """
You are a farming planning assistant who helps users optimize their farming plans.
Device a plan for the user based on the information they provide.
"""
# Your goal is to check if all_details_given is True. If it is, you will output the information in the required format.
# Else you'll ask the user for any missing information if necessary. Tell the user what information they need to provide still.


info_gathering_agent = Agent(
    model,
    result_type=FarmDetails,
    system_prompt=system_prompt,
    retries=2
)