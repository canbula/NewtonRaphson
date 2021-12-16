import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class NewtonRaphson(FuncAnimation):
    def __init__(self, f, x, h, iteration_limit):
        self.f = f
        self.x = x
        self.h = h
        self.iteration_limit = iteration_limit
        matplotlib.use("TkAgg")
        plt.rcParams["figure.figsize"] = [24, 18]
        self.fig, self.ax = plt.subplots()
        super(NewtonRaphson, self).__init__(self.fig, self.update_animation, interval=500)
        plt.show()

    def update_animation(self, frame):
        if frame > self.iteration_limit:
            quit()
        self.x = self.next_root()
        self.ax.clear()
        self.ax.set_title(f"Newton Raphson Method: Iteration={frame} @ x={self.x:.6f}")

    def diff(self):
        return (self.f(self.x+self.h) - self.f(self.x))/self.h

    def next_root(self):
        return self.x - self.f(self.x)/self.diff()

