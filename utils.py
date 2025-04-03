from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import os
import ollama

load_dotenv()

# class OllamaModel:
#     def __init__(self, model_name):
#         self.model_name = model_name

#     def generate(self, prompt):
#         return ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
    
def get_model():
    provider = os.getenv('PROVIDER', 'openai')
    llm = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')

    model = OpenAIModel(
        model_name = llm,
        base_url=base_url,
        api_key=api_key
    )

    return model
