from typing import List

import nltk
from nltk import RegexpParser
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

                
class Question2NounPhraseQueries:

    def __init__(self):
        pass

    def parse_question(self, q: str) -> List[str]:

        '''
        Parses natural language question to a string suitable for querying a full text search index
        by breaking it into noun phrase chunks. Each noun phrase becomes a separate query to the index.
        Runs resulting phrases through the same text analyser used to create the index.

        Parameters
        ----------
        q : str
            question to be parsed

        Returns
        -------
        list
            list of parsed query strings
        '''

        words = word_tokenize(q)
        words = [w for w in words if w not in ["’", "s"]]
        pos = pos_tag(words)

        p = '''
        NP: {<VBN|JJ.*>*<IN|CC>?<NN.*>+<IN|CC>?}
        <IN|CC>{}<NN.*|VBN>
        '''
        cp = RegexpParser(p)
        chunked = cp.parse(pos)

        searches = []
        for chunk in chunked.subtrees(filter = lambda t: (t.label() == "NP") or (t.label().startswith("VB"))):
            search = []
            for word in chunk.leaves():
                search.append(word[0])
            searches.append(search)

        searches = [' '.join(search) for search in searches]

        searches = list(set(searches))

        return searches