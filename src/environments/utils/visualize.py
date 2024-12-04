import robosuite as suite
import numpy as np

from src.utils.real_time import RealTimeHandler


def visualize_static(
        env : str,
        robots : str | list[str],
        time : float = 10.0
):
    """
    Visualize the environment with static robots.
    :param env: string representing the environment
    :param robots: robot(s) to use in the environment
    :param time: time to visualize the environment in seconds
    :return:
    """
    env = suite.make(
        env_name=env,
        robots=robots,
        has_renderer=True,
        render_camera=None,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    # reset the environment
    obs = env.reset()
    print(obs)

    # target real time rendering
    dt = 1.0 / env.control_freq
    n = int(time / dt)
    rt = RealTimeHandler(env.control_freq)
    rt.reset()
    for _ in range(n):
        action = np.zeros_like(env.action_spec[0])
        _ = env.step(action)  # take action in the environment
        env.render()  # render on display
        rt.sleep()