import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class NewtonRaphson(FuncAnimation):
    def __init__(self, func, x, h, tolerance, iteration_limit, x_limits, verbose=False):
        self.f = func
        self.x = x
        self.h = h
        self.tolerance = tolerance
        self.iteration_limit = iteration_limit
        self.x_limits = x_limits
        self.verbose = verbose
        matplotlib.use("TkAgg")
        plt.rcParams["figure.figsize"] = [24, 18]
        self.fig, self.ax = plt.subplots()
        super(NewtonRaphson, self).__init__(self.fig, self.update_animation,
                                            init_func=self.init_animation, interval=500)
        plt.show()

    def init_animation(self):
        pass

    def update_animation(self, frame):
        if np.abs(self.f(self.x)) < self.tolerance or frame >= self.iteration_limit:
            plt.close()
        self.ax.clear()
        self.ax.set_title(f"Newton Raphson Method: Iteration={frame} @ x={self.x:.6f}")
        x_points = np.linspace(self.x_limits[0], self.x_limits[1], self.x_limits[2])
        self.ax.set_xlim(self.x_limits[0], self.x_limits[1])
        self.ax.set_ylim(self.f(self.x_limits[0]), self.f(self.x_limits[1]))
        self.ax.plot(x_points, self.f(x_points), linewidth=2, color="red")
        self.ax.plot(x_points, [0 for _ in x_points], linewidth=1, color="black")
        self.ax.plot([self.x, self.x], [0, self.f(self.x)], linewidth=1, color="blue")
        self.ax.plot(self.x, self.f(self.x), marker="o", markersize=10)
        self.ax.plot([self.x, self.next_root()], [self.f(self.x), 0], linewidth=1, color="green")
        self.x = self.next_root()
        if self.verbose:
            print(f"Iteration={frame} @ x={self.x:.6f}")

    def diff(self):
        return (self.f(self.x+self.h) - self.f(self.x))/self.h

    def next_root(self):
        return self.x - self.f(self.x)/self.diff()


def f1(x):
    return x**2 - 4*x - 7


def f2(x):
    return 27*x**3 - 3*x + 1


def f3(x):
    return x**3 - x - 1


def main():
    x = 9
    h = 1e-6
    tolerance = 1e-3
    iteration_limit = 10
    x_limits = [2, 8, 100]
    NewtonRaphson(f1, x, h, tolerance, iteration_limit, x_limits, True)
    x = 5
    h = 1e-6
    tolerance = 1e-6
    iteration_limit = 30
    x_limits = [-1, 1, 100]
    NewtonRaphson(f2, x, h, tolerance, iteration_limit, x_limits, True)
    x = 1
    h = 1e-6
    tolerance = 1e-3
    iteration_limit = 10
    x_limits = [-0.5, 3, 100]
    NewtonRaphson(f3, x, h, tolerance, iteration_limit, x_limits, True)


if __name__ == "__main__":
    main()
