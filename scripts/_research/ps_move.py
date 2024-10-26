from transforms3d.euler import quat2euler

from src.demonstration.teleoperation.psmove.core.psmove_state import PsMoveState, PSMoveTarget

from src.demonstration.teleoperation.psmove.core.psmove_interface import PSMoveInterface

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

NUM_ELEMENTS = 20

class PsMovePlotter(PSMoveInterface):

    def __init__(self):
        super().__init__()

        self._app = QApplication([])

        self._pg_layout = pg.GraphicsLayoutWidget()
        self._pg_layout.setWindowTitle("PSMove Position and Orientation Data")
        self._pg_layout.resize(1200, 800)
        self._pg_layout.setBackground("w")

        self._plot_pos_left = self._pg_layout.addPlot(title="Left Controller Position", row=0, col=0, background="w")
        self._plot_euler_left = self._pg_layout.addPlot(title="Left Controller Orientation", row=1, col=0)

        self._plot_pos_right = self._pg_layout.addPlot(title="Right Controller Position", row=0, col=1)
        self._plot_euler_right = self._pg_layout.addPlot(title="Right Controller Orientation", row=1, col=1)

        self._left_t_vals = np.array([])
        self._right_t_vals = np.array([])

        self._pos_left_vals_x = np.array([])
        self._pos_left_vals_y = np.array([])
        self._pos_left_vals_z = np.array([])
        self._euler_left_vals_x = np.array([])
        self._euler_left_vals_y = np.array([])
        self._euler_left_vals_z = np.array([])

        self._pos_right_vals_x = np.array([])
        self._pos_right_vals_y = np.array([])
        self._pos_right_vals_z = np.array([])
        self._euler_right_vals_x = np.array([])
        self._euler_right_vals_y = np.array([])
        self._euler_right_vals_z = np.array([])

        self._pos_left_x_line = self._plot_pos_left.plot([], [], pen="red", label="x")
        self._pos_left_y_line = self._plot_pos_left.plot([], [], pen="blue", label="y")
        self._pos_left_z_line = self._plot_pos_left.plot([], [], pen="green", label="z")
        self._euler_left_x_line = self._plot_euler_left.plot([], [], pen="red", label="x")
        self._euler_left_y_line = self._plot_euler_left.plot([], [], pen="blue", label="y")
        self._euler_left_z_line = self._plot_euler_left.plot([], [], pen="green", label="z")

        self._pos_right_x_line = self._plot_pos_right.plot([], [], pen="red", label="x")
        self._pos_right_y_line = self._plot_pos_right.plot([], [], pen="blue", label="y")
        self._pos_right_z_line = self._plot_pos_right.plot([], [], pen="green", label="z")
        self._euler_right_x_line = self._plot_euler_right.plot([], [], pen="red", label="x")
        self._euler_right_y_line = self._plot_euler_right.plot([], [], pen="blue", label="y")
        self._euler_right_z_line = self._plot_euler_right.plot([], [], pen="green", label="z")

        self._pg_layout.show()

        self._t_left = 0
        self._t_right = 0

    def show(self):
        self._app.exec()

    def _update_plots_left(self, pos_left, quat_left):
        x, y, z = pos_left
        x_euler, y_euler, z_euler = quat2euler(quat_left)

        self._left_t_vals = np.append(self._left_t_vals[-NUM_ELEMENTS:], self._t_left)

        self._pos_left_vals_x = np.append(self._pos_left_vals_x[-NUM_ELEMENTS:], x)
        self._pos_left_vals_y = np.append(self._pos_left_vals_y[-NUM_ELEMENTS:], y)
        self._pos_left_vals_z = np.append(self._pos_left_vals_z[-NUM_ELEMENTS:], z)

        self._euler_left_vals_x = np.append(self._euler_left_vals_x[-NUM_ELEMENTS:], x_euler)
        self._euler_left_vals_y = np.append(self._euler_left_vals_y[-NUM_ELEMENTS:], y_euler)
        self._euler_left_vals_z = np.append(self._euler_left_vals_z[-NUM_ELEMENTS:], z_euler)

        self._pos_left_x_line.setData(self._left_t_vals, self._pos_left_vals_x)
        self._pos_left_y_line.setData(self._left_t_vals, self._pos_left_vals_y)
        self._pos_left_z_line.setData(self._left_t_vals, self._pos_left_vals_z)

        self._euler_left_x_line.setData(self._left_t_vals, self._euler_left_vals_x)
        self._euler_left_y_line.setData(self._left_t_vals, self._euler_left_vals_y)
        self._euler_left_z_line.setData(self._left_t_vals, self._euler_left_vals_z)

        self._t_left += 1

    def _update_plots_right(self, pos_right, quat_right):
        x, y, z = pos_right
        x_euler, y_euler, z_euler = quat2euler(quat_right)

        self._right_t_vals = np.append(self._right_t_vals[-NUM_ELEMENTS:], self._t_right)

        self._pos_right_vals_x = np.append(self._pos_right_vals_x[-NUM_ELEMENTS:], x)
        self._pos_right_vals_y = np.append(self._pos_right_vals_y[-NUM_ELEMENTS:], y)
        self._pos_right_vals_z = np.append(self._pos_right_vals_z[-NUM_ELEMENTS:], z)

        self._euler_right_vals_x = np.append(self._euler_right_vals_x[-NUM_ELEMENTS:], x_euler)
        self._euler_right_vals_y = np.append(self._euler_right_vals_y[-NUM_ELEMENTS:], y_euler)
        self._euler_right_vals_z = np.append(self._euler_right_vals_z[-NUM_ELEMENTS:], z_euler)

        self._pos_right_x_line.setData(self._right_t_vals, self._pos_right_vals_x)
        self._pos_right_y_line.setData(self._right_t_vals, self._pos_right_vals_y)
        self._pos_right_z_line.setData(self._right_t_vals, self._pos_right_vals_z)

        self._euler_right_x_line.setData(self._right_t_vals, self._euler_right_vals_x)
        self._euler_right_y_line.setData(self._right_t_vals, self._euler_right_vals_y)
        self._euler_right_z_line.setData(self._right_t_vals, self._euler_right_vals_z)

        self._t_right += 1


    def _on_update(self, state : PsMoveState) -> None:
        if state.target == PSMoveTarget.LEFT:
            self._update_plots_left, args=(state.pos, state.quat)

        elif state.target == PSMoveTarget.RIGHT:
            self._update_plots_right(state.pos, state.quat)




if __name__ == "__main__":
    plotter = PsMovePlotter()
    plotter.start()
    plotter.show()






