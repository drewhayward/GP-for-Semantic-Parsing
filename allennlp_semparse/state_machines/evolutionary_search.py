from collections import defaultdict
from typing import Dict, List, Tuple, TypeVar
from allennlp.data.vocabulary import Vocabulary

import torch

from allennlp_semparse.state_machines import util
from allennlp_semparse.state_machines.states import State
from allennlp_semparse.state_machines.beam_search import BeamSearch
from allennlp_semparse.state_machines.transition_functions import TransitionFunction

StateType = TypeVar("StateType", bound=State)

@BeamSearch.register('evolutionary-search')
class EvolutionarySearch(BeamSearch):
    def __init__(
        self,
        vocab: Vocabulary,
    ) -> None:
        self.vocab = vocab

    def single_evo_search(self):
        pass

    def search(
        self,
        num_steps: int,
        initial_state: StateType,
        transition_function: TransitionFunction,
        keep_final_unfinished_states: bool = True,
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
        # TODO:
        #   - need to get access to the world here so we can build the random search
        #   - Once we have a program we can use the world to translate it into an action
        #       sequence
        #   - Then we can use the vocabulary to convert that sequence into action indices
        #   - Finally we can take that fixed sequence and step through it to get the final score

        # valid actions stored in initial_state.grammar_state: List[GrammarState], 
        # for gs in that list the actions are in gs._valid_actions

        # self.vocab.get_token_from_index(39, namespace="rule_labels") -> 'List[Row] -> [<List[Row],ComparableColumn:List[Row]>, List[Row], ComparableColumn]'
        # transition_function.take_step(initial_state, allowed_actions=[{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}])
        best_states: Dict[int, List[StateType]] = {}
        return best_states
