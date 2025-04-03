from pydantic_ai import Agent
import logfire
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_model

logfire.configure(send_to_logfire='if-token-present')

model = get_model()

system_prompt = """
You are a agriculture and sustainable farming expert helping people plan their perfect agriculture system.

You will be given details of sustainability optimization, ehanced farming advise, and 
market research and recommendations, and it's your job to take all
of that information and summarize it in a neat final package to give to the user as your
final recommendation for their goal.
"""

final_planner_agent = Agent(model, system_prompt=system_prompt)