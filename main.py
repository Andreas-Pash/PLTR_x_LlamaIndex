from constants import *

from src.helper_functions import *
from src.llama_index_graph_rag_extractor import GraphRAGExtractor
from src.graph_rag_store import GraphRAGStore

from dotenv import load_dotenv
from azure.identity import ClientSecretCredential, DefaultAzureCredential, get_bearer_token_provider  

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def load_environment_variables():
    """Load required environment variables from local script or .env"""
    load_dotenv(dotenv_path="azure_envars.env", override=True)

def create_graph_index(docs_dir= "data/parsed_docs", num_nodes = None, save_index = True, index_path = None, save_kg = False, kg_path = None): 
    load_environment_variables()
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    
    docs = load_markdown_as_documents(root_dir= docs_dir)
    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=50,
        )
    nodes = splitter.get_nodes_from_documents(docs)
 
 
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

    
    extractor = GraphRAGExtractor( llm= llm,
                extract_prompt= triplets_extract_prompt,
                parse_fn= parse_entities_relationships,
                num_workers= async_workers,
                max_paths_per_chunk= max_knowledge_triplets
            )

    nodes_sample = nodes if num_nodes is None else nodes[:num_nodes]
    
    index = PropertyGraphIndex(
        nodes= nodes_sample,
        property_graph_store= GraphRAGStore(),
        kg_extractors=[extractor],
        show_progress= True
    )

    if save_index:
        save_path = index_path if index_path else os.path.join(os.getcwd(), "extracted_data/palantir_10k_graph")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        index.storage_context.persist(persist_dir= save_path)
    if save_kg:
        save_path = kg_path if kg_path else os.path.join(os.getcwd(), "extracted_data/kg_llm.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        index.property_graph_store.save_networkx_graph(name= save_path)

    return index

if __name__ == "__main__":
    create_graph_index()
