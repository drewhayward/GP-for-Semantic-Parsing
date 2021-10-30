from allennlp_semparse.dataset_readers.wikitables import WikiTablesDatasetReader
from allennlp_semparse.domain_languages.wikitables_language import WikiTablesLanguage
from allennlp_semparse.domain_languages.domain_language import nltk_tree_to_logical_form
from nltk import Tree
import random
from typing import Union, List

def get_children(action: str) -> List[str]:
    """
    Returns the list of productions from a production string.
    Ex
    "A -> B" would return ['B']
    "A -> [B, C]" would return ['B', 'C']
    """
    _, args = action.split(' -> ')
    if args.startswith('[') and args.endswith(']'):
        args = args[1:-1].split(', ')
    else:
        args = [args]

    return args

def random_query(lang, max_depth=5) -> Tree:
    productions = lang.get_nonterminal_productions()
    productions = {
        non_term: [get_children(op) for op in options]
        for non_term, options in productions.items()
    }

    def _random_query(non_term, depth=0) -> Tree:
        # Terminal expressions
        if non_term not in productions:
            return Tree(non_term, [])

        # We need to stop the branch so only selecting from options with 1 child
        shrinking_actions = [prod for prod in productions[non_term] if len(prod) == 1] 
        if depth >= max_depth and len(shrinking_actions) > 0:
            action = random.choice(shrinking_actions)
        else:
            action = random.choice(productions[non_term])


        if len(action) == 1:
            return _random_query(action[0], depth + 1)
        else:
            return Tree(non_term, [_random_query(child, depth+1) for child in action])

    return _random_query('@start@')

if __name__ == "__main__":
    reader = WikiTablesDatasetReader('./data/WikiTableQuestions', 'tmp/offline_search_output')

    data = reader.read('./data/WikiTableQuestions/data/random-split-1-train.examples')

    # Language
    example = next(data)
    lang = example['world'].metadata
    for _ in range(10):
        query = random_query(lang, max_depth = 3)
        print(nltk_tree_to_logical_form(query))
