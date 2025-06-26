import os
import re
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd 

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
        
        report_type = (md_path.parent.relative_to(root_dir).parts[1]) # 10K or 8K
        doc_timestamp = (md_path.parent.relative_to(root_dir).parts[2])  
        
        if report_type == "10K":
            doc_timestamp = doc_timestamp.split(" ")[0]
            period = 'Yearly'
        elif report_type == "10Q":
            doc_timestamp = doc_timestamp.replace("_","-")
            period = 'Quartelrly'
        elif report_type == "8K":
            doc_timestamp = doc_timestamp.replace("_","-")
            period = ' '
        else:
            logging.error(f"Invalid report type: {report_type}")
            raise ValueError(f"Invalid report type: {report_type}")

        metadata = {
            "filename": f"{parent_name}.pdf",
            "company": 'Palantir Technologies',
            "document_type": f"{report_type} report",
            "timestamp": doc_timestamp,
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
# # one(global) limiter for all OpenAI traffic

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

def kg_relations_to_df(index: PropertyGraphIndex, filename:str = None,  save_as_csv: bool = False):
    graph = index.property_graph_store.graph  
    first_edge = next(iter(graph.relations.values()))
    property_keys = list(first_edge.model_dump().get("properties", {}).keys())

    triples = [
        (
            edge.model_dump().get("source_id"),
            edge.model_dump().get("label"),
            edge.model_dump().get("target_id"),
            *[
                edge.model_dump().get("properties", {}).get(key)
                for key in property_keys if key!= 'triplet_source_id'
            ]
        )
        for _, edge in graph.relations.items()
    ]
    df = pd.DataFrame(triples, columns= ["subject", "predicate", "object"] + [key for key in property_keys if key!= 'triplet_source_id'] )
    
    if save_as_csv:
        path = filename if filename else os.path.join(os.getcwd(), "extracted_data/kg_relationships/kg_relations.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False) 

    return df
