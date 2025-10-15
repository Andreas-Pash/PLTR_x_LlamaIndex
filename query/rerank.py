from typing import List, Dict, Any, Tuple

from sentence_transformers import CrossEncoder, SentenceTransformer, util


class SBERTRerank:

    '''
    for semantic reranking of full text search results.
    or can be used for semantic search only.
    results of full text search must be combined at the question level first
    currently only contain methods for using SBERT models 
    
    sbert_model_mapping = [
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder"), <- best performing out the box cross encoder
        ("cross-encoder/qnli-electra-base", "cross-encoder"),
        ("cross-encoder/stsb-roberta-large", "cross-encoder"),
        ("msmarco-distilbert-cos-v5", "cos-sim"),
        ("msmarco-MiniLM-L12-cos-v5", "cos-sim"),
        ("msmarco-bert-base-dot-v5", "dot-score"),
        ("msmarco-distilbert-dot-v5", "dot-score") -> best performing out the box embeddings model
    ]  
    '''

    def __init__(self, model_name, mode):
        self.model_name = model_name
        self.mode = mode
        self.model = None
        self.load_model()

    
    def load_model(self):
        if self.mode == "cross-encoder":
            self.model = CrossEncoder(self.model_name)
        elif self.mode in ["cos-sim", "dot-score"]:
            self.model = SentenceTransformer(self.model_name)


    def rerank(self, ft_scores, docs: List[Dict[str, Any]], search_fields = List[str]) -> Dict[str, List[Tuple[str, float, int]]]:

        '''
        Apply chosen SBERT model and re-ranked results.
        Returns data structure required to run back through FTQuestionScoring.

        Parameters
        ----------
        ft_scores : FTQuestionScoring
            Object resulting from scoring of full text search results
        docs : List
            Indexed documents for embedding

        Returns
        -------
        dict : {
            "question": [
                ("doc_id", similarity score, new rank),
            ]
        }
        '''

        reranked = {}
        docs_scored = []
        for q, results in ft_scores.question_results.items():
            for r in results:
                target_doc = [d for d in docs if d["id"] == r[0]]
                fields = [v for k, v in target_doc[0].items() if k in search_fields]
                text = " ".join(fields)
                
                if self.mode == "cross-encoder":
                    score = self.sbert_xencoder_rerank(q, text)
                    docs_scored.append((r[0], score))
                elif self.mode == "cos-sim":
                    score = self.sbert_asym_cos_rerank(q, text)
                    docs_scored.append((r[0], score))
                elif self.mode == "dot-score":
                    score = self.sbert_asym_dot_rerank(q, text)
                    docs_scored.append((r[0], score))
                else:
                    print("model not available")

        docs_scored = sorted(docs_scored, key = lambda tup: tup[1], reverse = True)
        docs_reranked = []
        for idx, scored in enumerate(docs_scored):
            docs_reranked.append((scored[0], scored[1], idx))

        question = list(ft_scores.question_results.keys())
    
        reranked[question[0]] = docs_reranked

        return reranked

    

    def sbert_xencoder_rerank(self, query: str, result: str) -> float:
        '''
        Gets cross encoder score for question + search result document using SBERT model such as:
        - cross-encoder/ms-marco-MiniLM-L-6-v2
        - cross-encoder/qnli-electra-base
        - cross-encoder/stsb-roberta-large
        '''
        score = self.model.predict([query, result])

        return score


    def sbert_asym_cos_rerank(self, query: str, result: str) -> float:
        '''
        Embeds question and search result document using chosen SBERT models trained for asymmetric semantic search,
        and that return normalized embeddings suitable for cosine similarity. e.g.
        - msmarco-distilbert-cos-v5
        - msmarco-MiniLM-L12-cos-v5
        Favours retrieval of shorter documents.
        '''
        query_embedding = self.model.encode([query], convert_to_tensor = True)
        content_embedding = self.model.encode([result], convert_to_tensor = True)
        score = util.cos_sim(query_embedding, content_embedding)[0]
        score = score.numpy()[0]
    
        return score


    def sbert_asym_dot_rerank(self, query: str, result: str) -> float:
        '''
        Embeds question and search result document using chosen SBERT models trained fro asymmetric semantic search,
        and that produce embeddings of different length and so are only suitable for use with dot product. e.g.
        - msmarco-bert-base-dot-v5
        - msmarco-distilbert-dot-v5
        Favours retrieval of longer documents.
        '''

        query_embedding = self.model.encode([query], convert_to_tensor = True)
        content_embedding = self.model.encode([result], convert_to_tensor = True)
        score = util.dot_score(query_embedding, content_embedding)[0]
        score = score.numpy()[0]
    
        return score