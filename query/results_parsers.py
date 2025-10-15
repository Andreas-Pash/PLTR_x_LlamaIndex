def combine_query_results(results, k = 60):
    '''
    Uses reciprocal rank fusion to combine an re-rank the results from multiple queries to get a
    single result set per question.
    '''

    scores = {}
    for query, hits in results.items():
        for hit in hits:
            scores[hit[0]] = 0

    for query, hits in results.items():
        for hit in hits:
            rank = hit[2]
            scores[hit[0]] += 1 / (k + rank)

    scores = {k: v for k, v in sorted(scores.items(), key = lambda item: item[1], reverse = True)}

    combined_results = []
    rank = 0
    for doc, score in scores.items():
        combined_results.append((doc, score, rank))
        rank += 1

    return combined_results