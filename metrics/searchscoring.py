from typing import List
import numpy as np
from math import log2



class FTQuestionScoring:
    '''
    Score the results of a full text search at the question level.

    Attributes
        targets (list): list of targets doc ids
        question_results (dict): results output when querying the Index
        query_scores (list): list of FTQueryScoring objects for each query
        mrr (float): mean reciprocal rank - only valid if there is a single target
        m_ndcg (float): mean normalised discounted cumulative gain
    '''

    def __init__(self, targets: List[str], question_results) -> None:
        self.targets = targets
        self.question_results = question_results

        self.query_scores = []
        self.get_query_scores()

        self.mrr = None
        self.mean_reciprocal_rank()

        self.m_ndcg = None
        self.mean_ndcg()

    
    def get_query_scores(self) -> None:
        '''
        Get query level scores.
        '''

        for query, results in self.question_results.items():
            res = [x[0] for x in results]
            self.query_scores.append(FTQueryScoring(query, self.targets, res))

   
    def mean_reciprocal_rank(self) -> float:

        rrs = []
        for res in self.query_scores:
            rrs.append(res.rr)

        sum_rr = sum(rrs)
        if sum_rr == 0:
            mrr = 0
        else:
            mrr = sum_rr / len(rrs)

        self.mrr = float(mrr)

        return float(mrr)
    

    def mean_ndcg(self) -> float:

        ndcgs = []
        for res in self.query_scores:
            ndcgs.append(res.ndcg)

        sum_ndcg = sum(ndcgs)
        if sum_ndcg == 0:
            mndcg = 0
        else:
            mndcg = sum_ndcg / len(ndcgs)
        
        self.m_ndcg = float(mndcg)

        return float(mndcg)
    
  

class FTQueryScoring:
    '''
    Score results against each query as decomposed from the question.

    Attributes
        query (str): query string
        targets (list): list of targets doc ids
        results (list): list of doc ids returned by search
        recall (float): recall score
        rr (float): reciprocal rank - only valid if there is a single target
        ndcg (float): normalised discounted cumulative gain
    '''

    def __init__(self, query, targets, results):
        self.query = query
        self.targets = targets
        self.results = results

        self.recall = None
        self.score_recall()
        self.rr = None
        self.score_reciprocal_rank()
        self.ndcg = None
        self.score_ndcg()


    def score_recall(self) -> float:
        '''
        What proportion of correct results have been returned.
        
        recall = |relevant ∩ retrieved| / |relevant|

        Returns:
            float: The recall score, rounded to two decimal places.
                Returns 0.0 if there are no overlapping items between targets and results.
        '''

        t = set(self.targets)
        r = set(self.results)

        if len(t & r) == 0:
            self.recall = float(0)
        else:
            score = round(len(t & r) / float(len(t)), 2)
            self.recall = score
    

    def score_reciprocal_rank(self) -> float:
        '''
        The inverse of the rank of the first relevant document in a list of search results.
        
        RR = 1 / rank_of_first_relevant_result

        Returns:
            float: The reciprocal rank score. Returns 0.0 if no relevant item is found 
                    or if the targets list is empty.
        '''

        if len(self.targets) > 0:
            rank = float(self.results.index(self.targets[0])) + 1 if self.targets[0] in self.results else float(0)
            if rank == 0:
                self.rr = float(0)
            else:
                self.rr = float(1 / rank)
        else:
            self.rr = float(0)
        

    def score_ndcg(self) -> float:
        '''
        Normalised Discounted Cumulative Gain
        Measure of ranking quality that takes into account multiple targets.
        "Using a graded relevance scale of documents in a search-engine result set,
        DCG sums the usefulness, or gain, of the results discounted by their position
        in the result list. NDCG is DCG normalized by the maximum possible DCG of the
        result set when ranked from highest to lowest gain, thus adjusting for the
        different numbers of relevant results for different queries."
        '''

        t_scores = np.linspace(len(self.targets), 1, len(self.targets)).tolist()
        target_scores = [(x, y) for x, y in zip(self.targets, t_scores)]
        relevance = []
        for r in self.results:
            match = [t for t in target_scores if r == t[0]]
            if len(match) >= 1:
                relevance.append(match[0][1])
            else:
                relevance.append(0)

        dcg = sum(rel_k / log2(1 + k) for k, rel_k in enumerate(relevance, 1))

        in_results = set(self.results)
        in_targets = set(self.targets)
        diff = in_targets - in_results
        ideal_results = self.results + list(diff)

        ideal_relevance = []
        for r in ideal_results:
            match = [t for t in target_scores if r == t[0]]
            if len(match) >= 1:
                ideal_relevance.append(match[0][1])
            else:
                ideal_relevance.append(0)

        ideal_relevance = sorted(ideal_relevance, reverse = True)

        idcg = sum(rel_k / log2(1 + k) for k, rel_k in enumerate(ideal_relevance, 1))

        ndcg = dcg / idcg if idcg != 0 else 0

        self.ndcg = ndcg
