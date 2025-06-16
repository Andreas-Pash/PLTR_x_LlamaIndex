import os
import re
import logging
from pathlib import Path
from typing import List, Optional

import openai
import nest_asyncio
from dotenv import load_dotenv

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation
from llama_index.llms.azure_openai import AzureOpenAI

from tenacity import AsyncRetrying, retry_if_exception_type, wait_random_exponential, stop_after_attempt

from aiolimiter import AsyncLimiter




def load_markdown_as_documents(root_dir: str = "data") -> List[Document]:
    """
    Convert each parsed Markdown file into a llama-index Document.

    Metadata captured(dynamic ones):
        • year  – e.g. "2021"
        • period – "FY", "Q_{i}" for i in [1,4] (derived from the filename/parent)
    """

    documents: List[Document] = []

    for md_path in Path(root_dir).rglob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        parent_name = md_path.parent.name
        year: Optional[str] = None
        period: Optional[str] = None

        if " " in parent_name:
            year, period = parent_name.split(" ", 1)

        metadata = {
            "company": 'Palantir Technologies',
            "document_type": "Financial report",
            "year": year,
            "period": period,
        }
        documents.append(
            Document(
                text=text,
                metadata=metadata,
                doc_id=md_path.stem        
            )
        )
    return documents

def parse_entities_relationships(response_str: str):
        entity_pattern = r'\(entity_name:\s*(.*?),\s*entity_type:\s*(.*?),\s*entity_description:\s*(.*?)\)'
        relationship_pattern = r'\(source_entity:\s*(.*?),\s*target_entity:\s*(.*?),\s*relation:\s*(.*?),\s*relationship_description:\s*(.*?)\)'

        entities = re.findall(entity_pattern, response_str)
        relationships = re.findall(relationship_pattern, response_str)

        return entities, relationships

class SafeAzureOpenAI(AzureOpenAI):
    """
    Subclass of AzureOpenAI that adds automatic exponential backoff handling for HTTP 429 (rate limit) errors.

    This implementation is safe to use in asynchronous workflows and helps prevent failures due to temporary
    rate limiting by retrying requests with increasing delay intervals and optional jitter.

    For more on the backoff strategy, see:
    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    """

    async def _achat(self, messages, **kwargs):
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.Timeout)),
            wait=wait_random_exponential(multiplier=1, max=65),
            stop=stop_after_attempt(20),
            reraise=True,
        ):
            with attempt:
                return await super()._achat(messages, **kwargs)


###     ALTERNATIVE Limiter        ###
# # one limiter for all(i.e. global) OpenAI traffic

# RPM_THR = 60 
# limiter = AsyncLimiter(max_rate=RPM_THR, time_period=60) 

# def _wait_or_header(retry_state):
#     """
#     Custom Tenacity wait strategy:
#     1. If the exception has a Retry-After header, sleep exactly that +1 s.
#     2. Otherwise back-off with jitter: 0 – 2**attempt seconds (capped at 65 s).
#     """
#     exc = retry_state.outcome.exception()
#     if (
#         exc
#         and isinstance(exc, openai.RateLimitError)
#         and getattr(exc, "response", None) is not None
#     ):
#         ra = exc.response.headers.get("Retry-After")
#         if ra:
#             logging.info(f'Waiting {ra}s')
#             return float(ra) + 1          
 
#     attempt = retry_state.attempt_number
#     return min(61.0, random.uniform(0, 2 ** attempt))  

# class SafeAzureOpenAI(AzureOpenAI):   # Global limiter
#     @retry(
#         retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.Timeout)),
#         wait=_wait_or_header,
#         stop=stop_after_attempt(10),
#         reraise=True,
#     )
#     async def _achat(self, messages, **kwargs):
#         async with limiter:
#             return await super()._achat(messages, **kwargs) 


