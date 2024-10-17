import numpy as np

class CamConfig:
    """
    Class that holds basic camera properties
    """

    def __init__(
        self,
        azimuth : float = 200,
        elevation : float  = -20,
        lookat : np.ndarray = np.array([0, 0, 0]),
        distance : float = 2.5,
    ):
        self._azimuth = azimuth
        self._elevation = elevation
        self._lookat = lookat
        self._distance = distance


    @property
    def azimuth(self):
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth = value

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, value):
        self._elevation = value

    @property
    def lookat(self):
        return self._lookat

    @lookat.setter
    def lookat(self, value):
        if isinstance(value, np.ndarray) and value.shape == (3,):
            self._lookat = value
        else:
            raise ValueError("lookat must be a numpy array of shape (3,)")

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        if value > 0:
            self._distance = value
        else:
            raise ValueError("distance must be a positive value")