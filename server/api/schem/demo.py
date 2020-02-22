# encoding: utf-8

from pydantic import BaseModel
import numpy as np


class DemoInputX2Y(BaseModel):
    boundaryCondition: str
    equationConstraint: str
    model: str
    grid: list


class DemoOutputX2Y(BaseModel):
    heatmap: list = None

    def __init__(self):
        super(DemoOutputX2Y, self).__init__()

    def new_random_result(self):
        predmap = np.random.random((50, 50))
        femmap = np.random.random((50, 50))
        return predmap, femmap

    def gen_heatmap(self, predmap, femmap):
        x_grid, y_grid = np.meshgrid(range(50), range(50))
        heatmap = list(map(lambda x: np.expand_dims(x, 2), [x_grid, y_grid, predmap, femmap]))
        heatmap = np.concatenate(heatmap, axis=2).reshape(-1, 4)
        heatmap = heatmap.round(2)
        self.heatmap = heatmap.tolist()


if __name__ == "__main__":
    d = DemoOutputX2Y()
    p, pp =d.new_random_result()
    d.gen_heatmap(p, pp)


