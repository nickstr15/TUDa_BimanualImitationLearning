"""
This file implements a wrapper for recording episodes in an environment.
"""
import os
import imageio
from datetime import datetime
from robosuite.wrappers import Wrapper
import robosuite.macros as macros

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

class RecordingWrapper(Wrapper):
    def __init__(self, env):
        """
        Initialize the recording wrapper.

        Args:
            env (RobotEnv): The environment to wrap, must use use_camera_obs=True
        """
        super().__init__(env)

        # assert that the environment has camera observations
        assert self.env.use_camera_obs, "RecordingWrapper requires use_camera_obs=True"

        self.recording = False
        self.writer = None
        self.fps = self.env.control_freq
        self.path = None
        self.camera_name = None

    def start_recording(self, directory: str, camera_name: str = None):
        """
        Start recording the current episode

        Args:
            directory (str): Directory to save the recordings
            camera_name (str): Name of the camera to record, if None, the first camera found will be used
        """
        self.recording = True
        self.camera_name = camera_name if camera_name is not None else self.env.camera_names[0]

        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        env_name = self.env.__class__.__name__
        robots = "-".join([robot.name for robot in self.env.robots])
        file_name = f"{env_name}_{robots}_{time_stamp}.mp4"
        self.path = os.path.join(directory, file_name)
        #make sure the directory exists
        os.makedirs(directory, exist_ok=True)
        self.writer = imageio.get_writer(self.path, fps=self.fps)


    def stop_recording(self):
        """
        Stop recording the current episode
        """
        self.writer.close()
        print(f"Recording saved to {self.path}")
        self.writer = None
        self.recording = False
        self.path = None
        self.camera_name = None

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate visualization

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

        obs = ret[0]
        self.__process_observation(obs)

        return ret

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate visualization

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        obs = super().reset()
        self.__process_observation(obs)

        return obs

    def __process_observation(self, obs) -> None:
        """
        Process the observation and save the camera image if recording

        Args:
            obs (OrderedDict): The observation to process
        """
        if self.recording:
            frame = obs.get(self.camera_name + "_image", None)
            if frame is not None:
                self.writer.append_data(frame)
            else:
                self.writer.close()
                raise ValueError(f"Camera {self.camera_name} not found in observations")
