import numpy as np
from robosuite.wrappers.data_collection_wrapper import DataCollectionWrapper as RSDataCollectionWrapper
from typing_extensions import override


class DataCollectionWrapper(RSDataCollectionWrapper):
    """
    Custom data collection wrapper to handle non-flattened actions.
    """

    @override(RSDataCollectionWrapper)
    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super().step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.sim.get_state().flatten()
            self.states.append(state)

            ###########################################
            # CUSTOMIZED  #############################
            if not isinstance(action, np.ndarray):    #
                action = np.asarray(action.flatten()) #
            info = {"actions": action}                #
            ###########################################

            self.action_infos.append(info)

        # check if the demonstration is successful
        if self.env._check_success():
            self.successful = True

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret