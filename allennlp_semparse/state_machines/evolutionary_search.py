from copy import deepcopy
from collections import defaultdict, Counter
import random
from typing import Dict, List, Tuple, TypeVar

import torch
import matplotlib.pyplot as plt
from nltk import Tree as _Tree
from torch._C import _get_tracing_state

from allennlp_semparse.state_machines import util
from allennlp_semparse.state_machines import transition_functions
from allennlp_semparse.state_machines.states import State
from allennlp_semparse.state_machines.beam_search import Search, BeamSearch
from allennlp_semparse.state_machines.transition_functions import TransitionFunction
from allennlp_semparse.domain_languages.domain_language import nltk_tree_to_logical_form
from tqdm import trange

StateType = TypeVar("StateType", bound=State)

class Tree(_Tree):
    def __init__(self, *args, action=None):
        super().__init__(*args)
        self.action = action

    def __deepcopy__(self, memo):
        children_copy = [deepcopy(child) for child in self]
        return Tree(self._label, children_copy, action=self.action)

    def num_leaves(self):
        if len(self) == 0:
            return 1

        total = 0
        for c in self:
            total += c.num_leaves()
        return total
    
    def __eq__(self, other):
        return nltk_tree_to_logical_form(self) == nltk_tree_to_logical_form(other)

    def __hash__(self):
        return hash(nltk_tree_to_logical_form(self))

class Individual:
    tree: Tree
    score: float
    state: StateType
    gen_op: str
    
    def __init__(self, score: float, tree: Tree, state: StateType, attribution=None):
        self.tree = tree
        self.score = score
        self.state = state
        if attribution is not None:
            self.gen_op = attribution
        else:
            self.gen_op = 'init'
    
    def __eq__(self, other) -> None:
        return other.tree == self.tree

    def __hash__(self) -> int:
        return hash(self.tree)


def get_children(action: str, ret_non_term = False) -> List[str]:
    """
    Returns the list of productions from a production string.
    Ex
    "A -> B" would return ['B']
    "A -> [B, C]" would return ['B', 'C']
    """
    nonterm, args = action.split(' -> ')
    if args.startswith('[') and args.endswith(']'):
        args = args[1:-1].split(', ')
    else:
        args = [args]
    if ret_non_term:
        return nonterm, args
    else:
        return args

def action_seq_to_grammarstate(action_ids, init_state, trans_func):
    state = init_state
    for act_id in action_ids:
        state = trans_func.take_step(state, allowed_actions=[{act_id}])[0]
    return state

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

def mutate_leaf(program_tree, productions, ratio=1):
    mutated_tree = deepcopy(program_tree)
    if len(mutated_tree) == 0:
        return mutated_tree

    flat_nodes = flatten_tree(mutated_tree)
    num_leaves = sum(1 for n, _, _, _ in flat_nodes if len(n) == 0)

    for node, parent, id, _ in flat_nodes:
        if len(node) == 0 and random.random() < (1 / num_leaves) * ratio: # it's a leaf
            new_label = random.choice([p[0] for p in productions[parent.action[id]] if len(p) == 1])
            parent[id] = Tree(new_label, [])

    return mutated_tree

def mutate_subtree(program_tree, productions, ratio=1, max_depth=5):
    mutated_tree = deepcopy(program_tree)
    if len(mutated_tree) == 0:
        return mutated_tree

    flat_nodes = flatten_tree(mutated_tree)
    num_nodes = len(flat_nodes)

    for node, parent, id, depth in flat_nodes:
        if parent is not None and random.random() < (1 / num_nodes) * ratio: # it's a leaf
            parent[id] = random_subtree(node.label(), productions, max_depth - depth)

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

def tournament_selection(pop, k):
    tourney = random.sample(pop, k)

    return max(tourney, key=lambda x: x.score)

def tree_eq(t1, t2):
    return nltk_tree_to_logical_form(t1) == nltk_tree_to_logical_form(t2)

def action_seq_to_tree(action_sequence):
    curr_action, *remaining_actions = action_sequence

    def _node_from_actions(node: Tree, remaining_actions):
        action = remaining_actions.pop(0)
        nonterm, children = get_children(action, ret_non_term=True)
        
        assert(node.label() == nonterm)
        # Save child type info so we can edit later
        node.action = children

        if len(children) != 1:

            for child_type in children:
                child_node = Tree(child_type, [])
                node.append(child_node)
                remaining_actions = _node_from_actions(child_node, remaining_actions)
        else:
            node.append(Tree(children[0], []))

        return remaining_actions
    nonterm, args = get_children(curr_action, ret_non_term=True)

    tree = Tree(args[0], [])
    _node_from_actions(tree, remaining_actions)
    return tree

@Search.register('evolutionary-search')
class EvolutionarySearch(Search):
    def __init__(
        self,
        num_generations: int = 50,
        pop_size: int = 100,
        pop_lambda: int = 10,
        init_tree_depth: int = 5,
        mutation_ratio: int = 1,
        tournament_k: int = 5
    ) -> None:
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.init_tree_depth = init_tree_depth
        self.mutation_ratio = mutation_ratio
        self.pop_lambda = pop_lambda
        self.tournament_k = tournament_k

        self.search_count = 0

        self.startup_search = BeamSearch(1)

    def eval_program(self, program_tree, world, trans_func, init_state, action_to_id):
        program_string = nltk_tree_to_logical_form(program_tree)
        action_seq = world.logical_form_to_action_sequence(program_string)

        # Convert actions to ids
        action_ids = [action_to_id[act] for act in action_seq]

        state = action_seq_to_grammarstate(action_ids, init_state, trans_func)

        return state.score[0].item(), state

    def single_evo_search(self, world, init_state, trans_func, action_to_id, seed_state = None):
        self.search_count += 1
        # Generate random population
        productions = world.get_nonterminal_productions()
        productions = {
            non_term: [get_children(op) for op in options]
            for non_term, options in productions.items()
        }
        pop = set()
        while len(pop) < self.pop_size:
            pop.add(random_query(productions, self.init_tree_depth))
        pop = list(pop)

        if seed_state is not None:
            pop.append(seed_state)

        def fitness(t, attribution=None):
            score, state = self.eval_program(t, world, trans_func, init_state,action_to_id)
            return Individual(score, t, state, attribution)

        # (tree, score, state) tuples
        pop = [fitness(p) for p in pop]
        pop_stats = defaultdict(list)
        pop_stats['ops/cross/mutate_leaf'] = []
        pop_stats['ops/mutate_subtree'] = []
        pop_stats['ops/init'] = []

        # for each generation
        for gen in trange(self.num_generations):
            pop = set(pop)
            while len(pop) < self.pop_size + self.pop_lambda:
                if random.random() < 0.3: # crossover
                    ind1 = tournament_selection(pop, self.tournament_k)
                    ind2 = tournament_selection(pop, self.tournament_k)

                    cind1, cind2 = crossover_trees(ind1.tree, ind2.tree)
                    cind1 = mutate_leaf(cind1, productions, self.mutation_ratio)
                    cind2 = mutate_leaf(cind1, productions, self.mutation_ratio)
                    if not tree_eq(ind1.tree, cind1):
                        pop.update([fitness(cind1, 'ops/cross/mutate_leaf'), fitness(cind2, 'ops/cross/mutate_leaf')])
                else: # mutation
                    ind = tournament_selection(pop, self.tournament_k)
                    mutated = mutate_subtree(ind.tree, productions, ratio=2, max_depth=self.init_tree_depth)
                    if not tree_eq(mutated, ind.tree): # enforce mutations because we are doing mu + lambda
                        pop.add(fitness(mutated, 'ops/mutate_subtree'))

            # To get a tree out of a state 's'
            # id_to_action = {i: a for a, i in action_to_id.items()}
            # action_seq = [id_to_action[n] for n in s.action_history[0]]
            # convert action seq to tree with language/world object
            pop = list(pop)
            pop.sort(key=lambda x: x.score)
            pop = pop[-self.pop_size:]
            
            # Update population statistics
            gen_stats = Counter((p.gen_op for p in pop))
            for k in pop_stats.keys():
                if not k.startswith('ops/'): continue
                if k in gen_stats:
                    pop_stats[k].append(gen_stats[k])
                else:
                    pop_stats[k].append(0)

            fits = [p.score for p in pop] 
            pop_stats['fitness/best'].append(max(fits))
            pop_stats['fitness/worst'].append(min(fits))
            pop_stats['fitness/avg'].append(sum(fits)/len(fits))

            pop_stats['inds/avg_depth'].append(sum((p.tree.height() for p in pop))/ self.pop_size)
            pop_stats['inds/avg_leaves'].append(sum((p.tree.num_leaves() for p in pop))/ self.pop_size)

        # Operation attribution plot
        plt.clf()
        plt.subplot(3,1,1)
        ops = list(k for k in pop_stats.keys() if k.startswith('ops/'))
        for k in ops:
            plt.plot(range(self.num_generations), pop_stats[k])
        plt.ylim(-1, self.pop_size)
        plt.legend(ops)

        plt.subplot(3,1,2)
        ops = list(k for k in pop_stats.keys() if k.startswith('fitness/'))
        for k in ops:
            plt.plot(range(self.num_generations), pop_stats[k])
        plt.ylim(-10, 0)
        plt.legend(ops)

        plt.subplot(3,1,3)
        ops = list(k for k in pop_stats.keys() if k.startswith('inds/'))
        for k in ops:
            plt.plot(range(self.num_generations), pop_stats[k])
        plt.legend(ops)


        plt.savefig(f'figs/{self.search_count}metrics.png')
        
        return [p.state for p in pop[-5:]]

    def get_seed_state(self, num_steps, initial_state, trans_func, action_to_id, world):
        id_to_action = {i: a for a, i in action_to_id.items()}
        search_state = self.startup_search.search(num_steps, initial_state, trans_func)[0]
        
        # To get a tree out of a state 's'
        # convert action seq to tree with language/world object
        action_seq = [id_to_action[n] for n in search_state[0].action_history[0]]
        return action_seq_to_tree(action_seq)

    def search(
        self,
        num_steps: int,
        initial_state: StateType,
        transition_function: TransitionFunction,
        keep_final_unfinished_states: bool = True,
        world = None,
        actions = None,
    ) -> Dict[int, List[StateType]]:
        assert len(world) == 1, "Evo search must have a batch size of 1"
        action_to_id = {act.rule: i for i, act in enumerate(actions[0])}
        world = world[0]
        
        # try:
        # seed_tree = self.get_seed_state(num_steps, initial_state, transition_function, action_to_id, world)
        top_k = self.single_evo_search(world, initial_state, transition_function, action_to_id, None)

        # self.vocab.get_token_from_index(39, namespace="rule_labels") -> 'List[Row] -> [<List[Row],ComparableColumn:List[Row]>, List[Row], ComparableColumn]'
        # transition_function.take_step(initial_state, allowed_actions=[{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}])
        best_states: Dict[int, List[StateType]] = {0: top_k}
        return best_states
        # except:
        #     print('Failed to search for the parse')
        #     return {}
