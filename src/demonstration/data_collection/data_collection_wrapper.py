import os
import time

import numpy as np
from gymnasium import Wrapper
from robosuite.wrappers.data_collection_wrapper import DataCollectionWrapper as RSDataCollectionWrapper
from typing_extensions import override


class DataCollectionWrapper(RSDataCollectionWrapper):
    """
    Custom data collection wrapper to handle non-flattened actions.
    """
    @override
    def _start_new_episode(self) -> None:
        """
        Bookkeeping to do at the start of each new episode.

        Like the robosuite version, but drop the MjSim interface.
        """

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # time steps in current episode
        self.t = 0
        self.has_interaction = False

        # save the task instance (will be saved on the first env interaction)
        self._current_task_instance_xml = self.env.get_xml()
        self._current_task_instance_state = np.array(self.env.get_state().flatten())

        self.env.reset()
        self.env.set_state_from_flattened(self._current_task_instance_state)

    @override
    def step(self, action) -> tuple:
        """
        Extends vanilla step() function call to accommodate data collection

        Like the robosuite version, but drop the MjSim interface
        and ensures that action is flattened before saving it.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = Wrapper.step(self, action)
        self.env.render()
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.get_state().flatten()
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

    def clean_up(self) -> None:
        """
        Clean up the directory
        """
        os.system("rm -r {}".format(self.directory))
        self.has_interaction = False
        print("DataCollectionWrapper: Deleted directory at {}".format(self.directory))

    def update_state(self) -> None:
        """
        Update the current task instance state.
        Call this if the environment state has been changed outside the wrapper,
        e.g. after custom reset.
        :return:
        """
        self._current_task_instance_state = np.array(self.env.get_state().flatten())