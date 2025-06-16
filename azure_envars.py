import os
 
import logging
from pathlib import Path 

from azure.identity import ClientSecretCredential 
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  

    
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


# credential = ClientSecretCredential(os.environ['AZURE_TENANT_ID'], 
#                                     os.environ['AZURE_CLIENT_ID'],
#                                     os.environ['AZURE_CLIENT_SECRET'] )
# token = credential.get_token("https://cognitiveservices.azure.com/.default") 
# openai_api_key  = token.token
# os.environ["OPENAI_API_KEY"] = openai_api_key 
