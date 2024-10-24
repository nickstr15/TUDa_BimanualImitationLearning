import threading
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from transforms3d.euler import quat2euler

from src.data.teleoperation.core.psmove_state import PsMoveState, PSMoveTarget

mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
import numpy as np

plt.ion()
from src.data.teleoperation.core.psmove_interface import PSMoveInterface

NUM_ELEMENTS = 20

class PsMovePlotter(PSMoveInterface):

    def __init__(self):
        super().__init__()

        self._fig, self._axs = plt.subplots(2, 2, figsize=(10, 10))

        self._axs[0, 0].set_title("Left Controller - Position")

        self._axs[1, 0].set_title("Left Controller - Orientation")

        self._axs[0, 1].set_title("Right Controller - Position")

        self._axs[1, 1].set_title("Right Controller - Orientation")

        self._line_pos_x_left, = self._axs[0, 0].plot([], [], color="red", label="x")
        self._line_euler_x_left, = self._axs[1, 0].plot([], [], color="red", label="x")
        self._line_pos_y_left, = self._axs[0, 0].plot([], [], color="green", label="y")
        self._line_euler_y_left, = self._axs[1, 0].plot([], [], color="green", label="y")
        self._line_pos_z_left, = self._axs[0, 0].plot([], [], color="blue", label="z")
        self._line_euler_z_left, = self._axs[1, 0].plot([], [], color="blue", label="z")

        self._line_pos_x_right, = self._axs[0, 1].plot([], [], color="red", label="x")
        self._line_euler_x_right, = self._axs[1, 1].plot([], [], color="red", label="x")
        self._line_pos_y_right, = self._axs[0, 1].plot([], [], color="green", label="y")
        self._line_euler_y_right, = self._axs[1, 1].plot([], [], color="green", label="y")
        self._line_pos_z_right, = self._axs[0, 1].plot([], [], color="blue", label="z")
        self._line_euler_z_right, = self._axs[1, 1].plot([], [], color="blue", label="z")

        #plot legend
        #self._axs[0, 0].legend()
        #self._axs[1, 0].legend()
        #self._axs[0, 1].legend()
        #self._axs[1, 1].legend()

        self._t_left = 0
        self._t_right = 0

        self._lock = threading.Lock()


    def _update_plots_left(self, pos_left, quat_left):
        with self._lock:
            x, y, z = pos_left
            x_euler, y_euler, z_euler = quat2euler(quat_left)

            self._line_pos_x_left.set_xdata(np.append(self._line_pos_x_left.get_xdata()[-NUM_ELEMENTS:], self._t_left))
            self._line_pos_x_left.set_ydata(np.append(self._line_pos_x_left.get_ydata()[-NUM_ELEMENTS:], x))

            self._line_pos_y_left.set_xdata(np.append(self._line_pos_y_left.get_xdata()[-NUM_ELEMENTS:], self._t_left))
            self._line_pos_y_left.set_ydata(np.append(self._line_pos_y_left.get_ydata()[-NUM_ELEMENTS:], y))

            self._line_pos_z_left.set_xdata(np.append(self._line_pos_z_left.get_xdata()[-NUM_ELEMENTS:], self._t_left))
            self._line_pos_z_left.set_ydata(np.append(self._line_pos_z_left.get_ydata()[-NUM_ELEMENTS:], z))

            self._line_euler_x_left.set_xdata(np.append(self._line_euler_x_left.get_xdata()[-NUM_ELEMENTS:], self._t_left))
            self._line_euler_x_left.set_ydata(np.append(self._line_euler_x_left.get_ydata()[-NUM_ELEMENTS:], x_euler))

            self._line_euler_y_left.set_xdata(np.append(self._line_euler_y_left.get_xdata()[-NUM_ELEMENTS:], self._t_left))
            self._line_euler_y_left.set_ydata(np.append(self._line_euler_y_left.get_ydata()[-NUM_ELEMENTS:], y_euler))

            self._line_euler_z_left.set_xdata(np.append(self._line_euler_z_left.get_xdata()[-NUM_ELEMENTS:], self._t_left))
            self._line_euler_z_left.set_ydata(np.append(self._line_euler_z_left.get_ydata()[-NUM_ELEMENTS:], z_euler))

    def _update_plots_right(self, pos_right, quat_right):
        with self._lock:
            x, y, z = pos_right
            x_euler, y_euler, z_euler = quat2euler(quat_right)

            self._line_pos_x_right.set_xdata(np.append(self._line_pos_x_right.get_xdata()[-NUM_ELEMENTS:], self._t_right))
            self._line_pos_x_right.set_ydata(np.append(self._line_pos_x_right.get_ydata()[-NUM_ELEMENTS:], x))

            self._line_pos_y_right.set_xdata(np.append(self._line_pos_y_right.get_xdata()[-NUM_ELEMENTS:], self._t_right))
            self._line_pos_y_right.set_ydata(np.append(self._line_pos_y_right.get_ydata()[-NUM_ELEMENTS:], y))

            self._line_pos_z_right.set_xdata(np.append(self._line_pos_z_right.get_xdata()[-NUM_ELEMENTS:], self._t_right))
            self._line_pos_z_right.set_ydata(np.append(self._line_pos_z_right.get_ydata()[-NUM_ELEMENTS:], z))

            self._line_euler_x_right.set_xdata(np.append(self._line_euler_x_right.get_xdata()[-NUM_ELEMENTS:], self._t_right))
            self._line_euler_x_right.set_ydata(np.append(self._line_euler_x_right.get_ydata()[-NUM_ELEMENTS:], x_euler))

            self._line_euler_y_right.set_xdata(np.append(self._line_euler_y_right.get_xdata()[-NUM_ELEMENTS:], self._t_right))
            self._line_euler_y_right.set_ydata(np.append(self._line_euler_y_right.get_ydata()[-NUM_ELEMENTS:], y_euler))

            self._line_euler_z_right.set_xdata(np.append(self._line_euler_z_right.get_xdata()[-NUM_ELEMENTS:], self._t_right))
            self._line_euler_z_right.set_ydata(np.append(self._line_euler_z_right.get_ydata()[-NUM_ELEMENTS:], z_euler))



    def _on_update(self, state : PsMoveState) -> None:
        if state.target == PSMoveTarget.LEFT:
            self._update_plots_left(state.pos, state.quat)
            self._t_left += 1

        elif state.target == PSMoveTarget.RIGHT:
            self._update_plots_right(state.pos, state.quat)
            self._t_right += 1

    def update(self):
        with self._lock:
            for ax in self._axs.flatten():
                ax.relim()
                ax.autoscale_view()

            self._fig.canvas.draw()
            self._fig.canvas.flush_events()



if __name__ == "__main__":
    plotter = PsMovePlotter()
    plotter.start()

    while plotter._running:
        plotter.update()
        time.sleep(0.1)






