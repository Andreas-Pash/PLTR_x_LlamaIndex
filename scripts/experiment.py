# Databricks notebook source
# MAGIC %sh
# MAGIC rm -rf ~/.local/share/pypoetry
# MAGIC rm -rf ~/.local/lib/python*/site-packages/poetry*
# MAGIC rm -rf ~/.local/bin/poetry
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### GraphRAG Retrieval over Palantir 10-K Reports
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC
# MAGIC This project leverages **GraphRAG (Graph-augmented Retrieval-Augmented Generation)** to enhance question answering and document exploration capabilities over **Palantir's 10-K SEC filings**. We build a **knowledge graph** from the reports to power deeper, context-aware retrieval and reasoning.
# MAGIC
# MAGIC  
# MAGIC
# MAGIC #### Data Pipeline
# MAGIC
# MAGIC - **Source**: Quarterly 10-K filings from Palantir Technologies (PLTR), extracted from EDGAR.
# MAGIC - **Parsing**: Cleaned, sectioned, and semantically annotated `SmolDocling`.
# MAGIC - **Graph Construction**: 
# MAGIC   - Entities: business segments, financial metrics, partnerships, risks, etc.
# MAGIC   - Relationships: dependencies, time-based trends, causal inferences.
# MAGIC
# MAGIC  
# MAGIC
# MAGIC #### Core Capabilities
# MAGIC
# MAGIC - **Entity-centric QA**
# MAGIC - **Time-aware comparisons**
# MAGIC - **Traceable answers**
# MAGIC
# MAGIC  
# MAGIC #### Integration Points
# MAGIC
# MAGIC - **Azure OpenAI**
# MAGIC - **Neo4j**
# MAGIC - **MLflow**
# MAGIC - **NetworkX**
# MAGIC - **LangChain / LLM**
# MAGIC
# MAGIC  
# MAGIC
# MAGIC #### Future Directions
# MAGIC
# MAGIC - Integrate implicit path extraction for hidden or inferred connections.
# MAGIC - Expand to 10-Q reports to increase 
# MAGIC - Enable hypothesis testing
# MAGIC - Neo4j Integration
# MAGIC - Query and visualize index.
# MAGIC - N-tuple Abstraction
# MAGIC - Add `ImplicitPathExtractor`
# MAGIC - Refine GraphRAGExtractor to increas MLflow Async Task Tracking Granularity
# MAGIC - Use `networkx` to Download and Interact with Knowledge Graphs 
# MAGIC - Implement graph operations to get some metrics such as:
# MAGIC   - Traversals, subgraph extraction, degree centrality, etc. 

# COMMAND ----------

# DBTITLE 1,Import modules
import re
import time
import asyncio
import logging
from pathlib import Path
from typing import Any, List, Callable, Optional, Union, Dict

import pandas as pd
import nest_asyncio
from IPython.display import Markdown, display
import openai

nest_asyncio.apply()

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.async_utils import run_jobs
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.indices.property_graph import PropertyGraphIndex

# COMMAND ----------

# DBTITLE 1,Configuring OpenAI params
import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  
from azure.identity import ClientSecretCredential 

os.environ['AZURE_TENANT_ID'] = dbutils.secrets.get(scope = "dsai-sc-test-opai-kv-1-scope", 
                                                    key = "bsi-azure-tenant-id")
os.environ['AZURE_CLIENT_ID'] = dbutils.secrets.get(scope = "dsai-sc-test-opai-kv-1-scope", 
                                                    key = "dsai-ht-test-datalakes-data-sp")
os.environ['AZURE_CLIENT_SECRET'] = dbutils.secrets.get(scope = "dsai-sc-test-opai-kv-1-scope", 
                                                        key = "dsai-sc-test-openai-secret")

os.environ['AZURE_OPENAI_ENDPOINT'] = dbutils.secrets.get(scope = "dsai-sc-test-opai-kv-1-scope", 
                                                   key = "azure-openai-endpoint" )
os.environ["OPENAI_API_VERSION"] =  "2024-12-01-preview" 
os.environ["OPENAI_API_TYPE"] = "azure"
 
 
credential = ClientSecretCredential(os.environ['AZURE_TENANT_ID'], 
                                    os.environ['AZURE_CLIENT_ID'],
                                    os.environ['AZURE_CLIENT_SECRET'] )
token = credential.get_token("https://cognitiveservices.azure.com/.default") 
openai_api_key  = token.token
os.environ["OPENAI_API_KEY"] = openai_api_key

token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
client = AzureOpenAI(
            api_version = os.environ.get('OPENAI_API_VERSION'),
            azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT'),
            azure_ad_token_provider = token_provider
        )

# COMMAND ----------

# DBTITLE 1,Load parsed data
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

# COMMAND ----------

# DBTITLE 1,Recursive Chunking
docs = load_markdown_as_documents()
splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=50,
)
nodes = splitter.get_nodes_from_documents(docs)

# COMMAND ----------

max_knowledge_triplets = 10
extract_prompt = f"""
                    -Goal-
                    Given a financial or earnings report, identify all entities and their entity types from the text and all relationships among the identified entities.
                    These might be financial metrics, product/platform activity, partnerships etc.

                    Given the text, extract up to {max_knowledge_triplets} high-quality entity-relation triplets.

                    -Steps-

                    1. Identify all entities. For each identified entity, extract the following information:
                    - entity_name: Capitalized name of the entity (e.g., Palantir, Gotham platform)
                    - entity_type: Type of entity (e.g., Company, Platform, Metric, Partner)
                    - entity_description: Concise description of what the entity is or does

                    Format each entity like:
                    (entity_name: Entity Name, entity_type: Entity Type, entity_description: Entity Description)

                    2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
                    - source_entity: Name of the source entity
                    - target_entity: Name of the target entity
                    - relation: The relationship between the two
                    - relationship_description: Reason or context behind this connection

                    Format each relationship like:
                    (source_entity: Source, target_entity: Target, relation: Relation, relationship_description: Description)

                    3. When finished, output all extracted entities and relationships clearly.

                """

# COMMAND ----------

logging.basicConfig(level=logging.INFO) 

class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples)
    and entity/relation descriptions from text.
    """
    client: AzureOpenAI
    triplets_extract_prompt: str 
    num_workers: int
    
    def __init__(
        self,
        client: AzureOpenAI,
        triplets_extract_prompt: Optional[str] = None,
        num_workers: int = 4,
    ) -> None:
        """Init params.""" 

        super().__init__(
            client = client,
            triplets_extract_prompt = extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            num_workers= num_workers,
        ) 
    @classmethod
    def parse_entities_relationships(self, response_str: str):
        entity_pattern = r'\(entity_name:\s*(.*?),\s*entity_type:\s*(.*?),\s*entity_description:\s*(.*?)\)'
        relationship_pattern = r'\(source_entity:\s*(.*?),\s*target_entity:\s*(.*?),\s*relation:\s*(.*?),\s*relationship_description:\s*(.*?)\)'

        entities = re.findall(entity_pattern, response_str)
        relationships = re.findall(relationship_pattern, response_str)

        return entities, relationships

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )
    
    async def _aextract(self, node: BaseNode) -> BaseNode:
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        messages = [
            {"role": "system", "content": self.triplets_extract_prompt},
            {"role": "user", "content": text}
        ]
        
        logging.info(f"Extracting triples from node content: {text[:50]}...")
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
            ))
            response_text = response.choices[0].message.content
            entities, relationships = self.parse_entities_relationships(response_text)
            # print(f'First 3 Entities are {entities[:3]}')
            # print(f'First 3 Relationships are {relationships[:3]}')

        except Exception as e:
            time.sleep(60) 
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",   #gpt-4.1-provex
                    messages=messages,
                    temperature=0,
                ))

                response_text = response.choices[0].message.content
                entities, relationships = self.parse_entities_relationships(response_text)
            except Exception as e:
                logging.error(f"Triplets extraction failed: {e}")
                entities = []
                relationships = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        
        metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            metadata["entity_description"] = description
            entity_node = EntityNode(name=entity, label=entity_type, properties=metadata)
            existing_nodes.append(entity_node)

        metadata = node.metadata.copy()
        for subj, obj, rel, description in relationships:
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )
            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node
    
    # async def _run_jobs(self, jobs, show_progress: bool = True):
    #     """Run jobs in parallel with semaphore limit."""
    #     sem = asyncio.Semaphore(self.num_workers)
    #     async def sem_job(job):
    #         async with sem:
    #             return await job

    #     return await asyncio.gather(*(sem_job(job) for job in jobs))
    
    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = True, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))
        
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

# COMMAND ----------

import re
import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.core.graph_stores import SimplePropertyGraphStore


class GraphRAGStore(SimplePropertyGraphStore):
    community_summary = {}
    max_cluster_size = 5

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        prompt =( "You are provided with a set of relationships from a knowledge graph, each represented as "
                "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                "relationships. The summary should include the names of the entities involved and a concise synthesis "
                "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                "integrates the information in a way that emphasizes the key aspects of the relationships."
                )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
         
        response = self.client.chat.completions.create(
                model="gpt-4o-mini",    # gpt-4.1-provex
                messages=messages,
                temperature=0,
            )
        response_text = response.choices[0].message.content
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()

        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))
        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """Collect detailed information for each node based on their community."""
        community_mapping = {item.node: item.cluster for item in clusters}
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in community_info:
                community_info[cluster_id] = []
                
            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail)
        return community_info

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary
 

# COMMAND ----------

# DBTITLE 1,LLM and embeddings model conf
from llama_index.llms.azure_openai import AzureOpenAI 
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

azure_openai_llm = AzureOpenAI(
    model='gpt-4.1',
    deployment_name= 'gpt-4.1-provex',
    azure_endpoint= os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key= os.environ['OPENAI_API_KEY'],
    api_version= os.environ['OPENAI_API_VERSION'],
    temperature=0,
)
# Settings.llm = azure_openai_llm  

embed_model = AzureOpenAIEmbedding(
    chunk_size=1000,
    model="text-embedding-ada-002",  
    api_key=openai_api_key,
) 
# Settings.embed_model =  client.embeddings  

# COMMAND ----------

extractor = GraphRAGExtractor(
                client=client,
                triplets_extract_prompt = extract_prompt,
                num_workers = 4
            )

index = PropertyGraphIndex(
    nodes=nodes[:3],
    property_graph_store=GraphRAGStore(),
    kg_extractors=[extractor],   
    llm=azure_openai_llm,  
    embed_model=embed_model,
    show_progress=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ TODO List
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🗂️ Modularization & Project Structure
# MAGIC   - Refactor notebook into a modular Python package.
# MAGIC
# MAGIC
# MAGIC #### 🔗 Neo4j Integration
# MAGIC - Connect to a Neo4j graph database using the `neo4j` Python driver to AuraDB
# MAGIC - Develop utility functions to:
# MAGIC   - Query and visualize subgraphs.
# MAGIC   - Import/export data to/from Neo4j and local structures (e.g., NetworkX).
# MAGIC - Enable support for Cypher queries from within the ML pipeline..
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📦 N-tuple Abstraction
# MAGIC - Define a flexible `n`-tuple data structure for representing knowledge graph entities and relationships 
# MAGIC - Extend to support:
# MAGIC   - Higher-order relationships (e.g., quads or n-tuples).
# MAGIC   - Temporal and contextual metadata.
# MAGIC - Provide transformation utilities between `n`-tuples and NetworkX / Neo4j graph formats.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔍 Add `ImplicitPathExtractor`
# MAGIC - Create a module/class to extract implicit paths or relationships
# MAGIC - Add to existing GraphRAGExtractor  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📈 Improve MLflow Async Task Tracking Granularity
# MAGIC - Refine GraphRAGExtractor to improve tracking to capture finer-grained details of async workflows
# MAGIC - (Possibly build a decorator or context manager to wrap around async functions.)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🌐 Use `networkx` to Download and Interact with Knowledge Graphs
# MAGIC - Download knowledge graphs from public sources or APIs.
# MAGIC - Implement graph operations to get some metrics such as:
# MAGIC   - Traversals, subgraph extraction, degree centrality, etc. 
# MAGIC
# MAGIC ---
# MAGIC

# COMMAND ----------

# DBTITLE 1,Explore relationships
graph = index.property_graph_store.graph
# Print edges with relationships
print("\n=== Relationships ===\n")
for i, edge in enumerate(graph.relations.items()):
    if i<10:
        print(edge)

# COMMAND ----------

from openai import AzureOpenAI
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings

oaiembeds =  AzureOpenAIEmbeddings( 
        chunk_size=1000,
        deployment = "text-embedding-ada-002" ,
        validate_base_url=True,
        openai_api_key= token.token
    )
oaiembeds.embed_documents(["hello world"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Neo4j Integration

# COMMAND ----------

import pandas as pd
triplets = []
graph = index.property_graph_store.graph 
for i, node in enumerate(graph.relations.items()):
    relations = node.metadata.get('relations', [])
    for relation in relations:
        subject = relation.source_id
        predicate = relation.label
        obj = relation.target_id
        triplets.append((subject, predicate, obj))

df = pd.DataFrame(triplets, columns=["subject", "predicate", "object"])
df.to_csv("triples.csv", index=False)

# COMMAND ----------

from neo4j import GraphDatabase

uri = "neo4j+s://ad9f52c8.databases.neo4j.io"
username = "neo4j"
password = "B5O0mYq5Va3JE-FvxiDUjkjlGYutu3jLopyollF65jE"
driver = GraphDatabase.driver(uri, auth=(username, password))

def create_graph(tx, subject, predicate, object_):
    query = f"""
    MERGE (s:Entity {{name: $subject}})
    MERGE (o:Entity {{name: $object}})
    MERGE (s)-[r:{predicate.upper().replace(" ", "_")}]->(o)
    """
    tx.run(query, subject=subject, object=object_)

with driver.session() as session:
    for s, p, o in triplets:
        session.write_transaction(create_graph, s, p, o)
