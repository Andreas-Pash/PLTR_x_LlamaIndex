# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# DBTITLE 1,Install dependancies
!pip install -r requirements.txt
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import modules
from constants import * 
import networkx as nx
import pyvis

from src.helper_functions import *
from src.llama_index_graph_rag_extractor import GraphRAGExtractor
from src.graph_rag_store import GraphRAGStore

# COMMAND ----------

# DBTITLE 1,Configuring OpenAI params
# Manage environment variables required to attach LLM service in this context
exec(open('azure_envars.py').read())

# COMMAND ----------

# DBTITLE 1,Load and chunk parsed data
from main import create_graph_index
save_idx_path = os.path.join(os.getcwd(), "outputs/palantir_8k_10k_graph")
save_kg_path = os.path.join(os.getcwd(), "outputs/8k_10k_kg.html")

index = create_graph_index(docs_dir= "data",
                           # num_nodes = 1,
                           save_index = True,
                           save_kg = True,
                           index_path = save_idx_path,
                           kg_path = save_kg_path
                        )
index.property_graph_store.build_communities()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ TODO List
# MAGIC
# MAGIC --- 
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

# COMMAND ----------

from azure.identity import ClientSecretCredential, DefaultAzureCredential, get_bearer_token_provider  
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
llm = SafeAzureOpenAI(
    model= MODEL_NAME,
    deployment_name= DEPLOYMENT_NAME,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider= token_provider,
    use_azure_ad= True,
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0,
)
Settings.embed_model = embed_model
Settings.llm = llm

# COMMAND ----------

# DBTITLE 1,Load index
from llama_index.core import load_index_from_storage, StorageContext
old_index = load_index_from_storage(storage_context=StorageContext.from_defaults(persist_dir="palantir_10k_graph"))

# COMMAND ----------

old_index = PropertyGraphIndex.from_existing(
    property_graph_store= index.property_graph_store  ,
    # optional, neo4j also supports vectors directly
    vector_store= index.vector_store,
    embed_kg_nodes=True,
)

# COMMAND ----------

from src.custom_query_engine import GraphRAGQueryEngine
 
graph_store = index.property_graph_store
query_engine = GraphRAGQueryEngine(graph_store= graph_store,
                                   llm=llm,
                                   index=old_index
                                ) 


query_str = "What was Palantir's revenue in 2022?"
response = query_engine.query(query_str)
response
 

# COMMAND ----------

nodes = index.as_retriever(
            similarity_top_k= 8
        ).retrieve(query_str)
nodes
 
 
 
# COMMAND ----------

from llama_index.core.indices.property_graph import (
    PGRetriever,
    VectorContextRetriever,
    LLMSynonymRetriever,
)

sub_retrievers = [
    VectorContextRetriever(index.property_graph_store,
                           ),
    LLMSynonymRetriever(index.property_graph_store,
                    ),
]

retriever = PGRetriever(index = index, sub_retrievers=sub_retrievers)
nodes = retriever.retrieve(query_str)
nodes
 
# COMMAND ----------

from src.helper_functions import kg_relations_to_df
kg_relations_to_df(index, save_as_csv = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Neo4j Integration

# COMMAND ----------

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password=" ",
    url=" ",
    )

# COMMAND ----------

from neo4j import GraphDatabase

uri = " "
username = "neo4j"
password = " "
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
