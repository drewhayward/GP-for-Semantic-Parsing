from collections import defaultdict
import random
from typing import Dict, List, Tuple, TypeVar

import torch
from allennlp.data.vocabulary import Vocabulary
from nltk import Tree

from allennlp_semparse.state_machines import util
from allennlp_semparse.state_machines.states import State
from allennlp_semparse.state_machines.beam_search import BeamSearch
from allennlp_semparse.state_machines.transition_functions import TransitionFunction
from allennlp_semparse.domain_languages.domain_language import nltk_tree_to_logical_form
from random_queries import random_query

StateType = TypeVar("StateType", bound=State)

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


@BeamSearch.register('evolutionary-search')
class EvolutionarySearch(BeamSearch):
    def __init__(
        self,
        vocab: Vocabulary,
        num_generations: int = 300,
        pop_size: int = 300,
        init_tree_depth: int = 5
    ) -> None:
        self.vocab = vocab
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.init_tree_depth = init_tree_depth

    def random_query(self, world, max_depth = 5) -> Tree:
        productions = world.get_nonterminal_productions()
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

    def eval_program(self, program_tree, world, trans_func, init_state, action_to_id):
        #   - need to get access to the world here so we can build the random search
        #   - Once we have a program we can use the world to translate it into an action
        #       sequence
        #   - Then we can use the vocabulary to convert that sequence into action indices
        #   - Finally we can take that fixed sequence and step through it to get the final score
        program_string = nltk_tree_to_logical_form(program_tree)
        action_seq = world.logical_form_to_action_sequence(program_string)

        # Convert actions to ids
        # TODO: Why don't the last columns get in the vocab correctly?
        action_ids = [self.vocab.get_token_index(act, namespace="rule_labels") for act in action_seq]

        state = init_state
        for act_id in action_ids:
            state = trans_func.take_step(state, allowed_actions=[{act_id}])[0]

        return state.score.item()



    def single_evo_search(self, world, init_state, trans_func):
        
        # Generate random population
        pop = [random_query(world, self.init_tree_depth) for _ in range(self.pop_size)]

        # for each generation
        for gen in range(self.num_generations):
            # Selection
            # Mutation/ Crossover step
            # Evaluate using the fitness function 
            # Keep pop size programs
            pass
        pass

    def search(
        self,
        num_steps: int,
        initial_state: StateType,
        transition_function: TransitionFunction,
        keep_final_unfinished_states: bool = True,
        world = None,
        actions = None,
    ) -> Dict[int, List[StateType]]:
        """
        Parameters
        ----------
        num_steps : ``int``
            How many steps should we take in our search?  This is an upper bound, as it's possible
            for the search to run out of valid actions before hitting this number, or for all
            states on the beam to finish.
        initial_state : ``StateType``
            The starting state of our search.  This is assumed to be `batched`, and our beam search
            is batch-aware - we'll keep ``beam_size`` states around for each instance in the batch.
        transition_function : ``TransitionFunction``
            The ``TransitionFunction`` object that defines and scores transitions from one state to the
            next.
        keep_final_unfinished_states : ``bool``, optional (default=True)
            If we run out of steps before a state is "finished", should we return that state in our
            search results?

        Returns
        -------
        best_states : ``Dict[int, List[StateType]]``
            This is a mapping from batch index to the top states for that instance.
        """
        assert len(world) == 1, "Evo search"
        action_to_id = {act.rule: i for i, act in actions[0]}
        world = world[0]
        

        # valid actions stored in initial_state.grammar_state: List[GrammarState], 
        # for gs in that list the actions are in gs._valid_actions

        # self.vocab.get_token_from_index(39, namespace="rule_labels") -> 'List[Row] -> [<List[Row],ComparableColumn:List[Row]>, List[Row], ComparableColumn]'
        # transition_function.take_step(initial_state, allowed_actions=[{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}])
        best_states: Dict[int, List[StateType]] = {}
        return best_states
