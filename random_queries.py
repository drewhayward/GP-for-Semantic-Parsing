from collections import defaultdict
from copy import deepcopy
from allennlp_semparse.dataset_readers.wikitables import WikiTablesDatasetReader
from allennlp_semparse.domain_languages.wikitables_language import WikiTablesLanguage
from allennlp_semparse.domain_languages.domain_language import nltk_tree_to_logical_form
from nltk import Tree as _Tree
import random
from typing import Union, List

class Tree(_Tree):
    def __init__(self, *args, action=None):
        super().__init__(*args)
        self.action = action

    def __deepcopy__(self, memo):
        children_copy = [deepcopy(child) for child in self]
        return Tree(self._label, children_copy, action=self.action)
        

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

def flatten_tree(program_tree: Tree):
    nodes = []

    frontier = [(program_tree, None, None, 0)]
    while frontier:
        state = frontier.pop()
        nodes.append(state)
        node, parent, id, depth = state

        for i, child in enumerate(node):
            frontier.append((child, node, i, depth + 1))

    return nodes

def mutate_leaf(program_tree, productions):
    mutated_tree = deepcopy(program_tree)
    if len(mutated_tree) == 0:
        return mutated_tree

    flat_nodes = flatten_tree(mutated_tree)
    num_leaves = sum(1 for n, _, _, _ in flat_nodes if len(n) == 0)

    for node, parent, id, _ in flat_nodes:
        if len(node) == 0 and random.random() < (1 / num_leaves): # it's a leaf
            new_label = random.choice([p[0] for p in productions[parent.action[id]] if len(p) == 1])
            parent[id] = Tree(new_label, [])

    return mutated_tree

def mutate_subtree(program_tree, productions):
    mutated_tree = deepcopy(program_tree)
    if len(mutated_tree) == 0:
        return mutated_tree

    flat_nodes = flatten_tree(mutated_tree)
    num_nodes = len(flat_nodes)

    for node, parent, id, depth in flat_nodes:
        if parent is not None and random.random() < (1 / num_nodes): # it's a leaf
            parent[id] = random_subtree(node.label(), productions, 5)

    return mutated_tree



def crossover_trees(tree1: Tree, tree2: Tree):
    def _get_nonterm_map(nodes):
        m = defaultdict(list)
        for state in nodes:
            node, parent, id, depth = state
            if parent is None: # Disallow crossover at the root
                continue
            if len(node) == 0: # Disallow crossover at leaf
                continue
            
            m[node.label()].append(state)

        return m
    # Make copies
    tree1 = deepcopy(tree1)
    tree2 = deepcopy(tree2)

    nodes1 = flatten_tree(tree1)
    nodes2 = flatten_tree(tree2)
    nonterms1 = _get_nonterm_map(nodes1)
    nonterms2 = _get_nonterm_map(nodes2)

    # Find common nonterminals
    common = list(set(nonterms1.keys()).intersection(nonterms2))

    if len(common) == 0:
        return tree1, tree2

    swapterm = random.choice(common)

    node1, parent1, id1, _ = random.choice(nonterms1[swapterm])
    node2, parent2, id2, _ = random.choice(nonterms2[swapterm])

    parent1[id1] = node2
    parent2[id2] = node1

    return tree1, tree2

def random_subtree(non_term, productions, max_depth, depth=0) -> Tree:
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
        return random_subtree(action[0], productions, max_depth, depth + 1)
    else:
        return Tree(
                non_term,
                [random_subtree(child, productions, max_depth, depth+1) for child in action],
                action=action)

def random_query(productions, max_depth=5) -> Tree:
    return random_subtree('@start@', productions, max_depth)

if __name__ == "__main__":
    reader = WikiTablesDatasetReader('./data/WikiTableQuestions', 'tmp/offline_search_output')

    data = reader.read('./data/WikiTableQuestions/data/random-split-1-train.examples')

    # Language
    example = next(data)
    lang = example['world'].metadata
    productions = lang.get_nonterminal_productions()
    productions = {
        non_term: [get_children(op) for op in options]
        for non_term, options in productions.items()
    }
    for _ in range(100):
        
        query1 = random_query(productions, max_depth = 3)
        query2 = random_query(productions, max_depth = 3)
        if len(query1) == 0 or len(query2) == 0:
            continue
        print(nltk_tree_to_logical_form(query1))
        print(nltk_tree_to_logical_form(query2))
        query1, query2 = crossover_trees(query1, query2)
        print(nltk_tree_to_logical_form(query1))
        print(nltk_tree_to_logical_form(query2))
        print('---')
