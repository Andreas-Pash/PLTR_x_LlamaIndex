OPENAI_API_VERSION = '2024-12-01-preview'
MODEL_NAME = 'gpt-4.1'
DEPLOYMENT_NAME = 'gpt-4.1-provex'

async_workers = 3

max_knowledge_triplets = 10
triplets_extract_prompt = """

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

-Real Data-
######################
text: {text}
######################
output:

"""
