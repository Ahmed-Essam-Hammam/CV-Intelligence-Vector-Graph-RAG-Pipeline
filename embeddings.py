from langchain_openai import AzureOpenAIEmbeddings
from config import AZURE_API_KEY, AZURE_ENDPOINT, EMBEDDING_MODEL


_embeddings_instance = None


def get_embeddings():
    global _embeddings_instance

    if _embeddings_instance is None:
        _embeddings_instance = AzureOpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version="2024-02-01"
        )

    return _embeddings_instance
