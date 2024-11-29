import numpy as np
import matplotlib.pyplot as plt
import robosuite as suite

import src.environments

def reset_loop():
    env = suite.make(
        "TwoArmPickPlace",
        robots=["Panda", "Panda"],
        env_configuration="parallel",
        has_renderer=True,
        camera_names="frontview",
        has_offscreen_renderer=True,
        use_object_obs=True,
        use_camera_obs=True,
    )

    env.deterministic_reset = False
    env.hard_reset = True

    hammer_positions_after_reset = []
    imgs_after_reset = []
    for i in range(4):
        obs = env.reset()
        img = obs["frontview_image"]
        hammer_pos = obs["hammer_pos"]

        hammer_positions_after_reset.append(hammer_pos)
        imgs_after_reset.append(img)

        # some dummy steps
        for _ in range(100):
            action = np.random.uniform(env.action_spec[0])*0.1
            _ = env.step(action)

        # needed to prevent freezing of the viewer
        env.viewer.close()

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    # print the hammer positions
    for i, (pos, img, ax) in enumerate(
            zip(hammer_positions_after_reset, imgs_after_reset, axs.flatten())
    ):
        ax.set_title(f"Hammer position after reset {i}\n{pos}")
        ax.imshow(img, origin="lower")

    plt.show()


if __name__ == "__main__":
    reset_loop()