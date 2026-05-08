from langchain_openai import AzureChatOpenAI
from config import AZURE_API_KEY, AZURE_ENDPOINT, LLM_MODEL



_llm_instance = None

def get_llm():
    global _llm_instance

    if _llm_instance is None:
        _llm_instance = AzureChatOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version="2024-12-01-preview",
            deployment_name=LLM_MODEL,
            temperature=0
        )

    return _llm_instance




# from langchain_openai import ChatOpenAI
# from config import CEREBRAS_API_KEY, LLM_MODEL

# def get_llm():
#     return ChatOpenAI(
#         base_url="https://api.cerebras.ai/v1",
#         model=LLM_MODEL,
#         api_key=CEREBRAS_API_KEY,
#         temperature=0
#     )