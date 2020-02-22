# encoding: utf-8

import numpy as np
import scipy.io as scio
from fenics import *


class MapGenerator(object):
    def __init__(self, length=0.1, grids=50, N=20, tol=1e-14):
        self.tol = tol
        self.length = length
        self.N = N
        self.grids = grids

    def build_f(self):
        def judge_machine(x):
            for ll in self.lst:
                ll_y = ll // 10
                ll_x = ll % 10
                if (self.length / 10 * ll_x <= x[0] <= self.length / 10 * (ll_x + 1)) and (
                        self.length / 10 * ll_y <= x[1] <= self.length / 10 * (ll_y + 1)):
                    return 1
            return 0

        class F(UserExpression):
            def eval(self, value, x):
                value[0] = 1e4 * judge_machine(x)

            def value_shape(self):
                return ()

        self.f = F()

    def build_boundary_D(self):
        def boundary_D(x, on_boundary):
            return on_boundary and (near(x[1], 0, self.tol) and (
                    self.length / 2 - self.length / 4 < x[0] < self.length / 2 + self.length / 4))

        self.boundary_D = boundary_D

    def build_g(self):
        self.g = Constant(0.)

    # lst: 1~100
    def gen_u(self):
        # Create mesh and define function space
        mesh = RectangleMesh(Point(0., 0.), Point(self.length, self.length), self.N - 1, self.N - 1)
        # mesh = UnitSquareMesh(8, 8)
        V = FunctionSpace(mesh, 'P', 1)

        self.build_boundary_D()
        self.build_g()
        self.build_f()

        # Define boundary condition
        u_D = Constant(298.)

        # def boundary_N(x, on_boundary):
        #     return on_boundary and near(x[1], 0, tol) and (l/2 - l/4 < x[0] < l/2 + l/4)

        bc_D = DirichletBC(V, u_D, self.boundary_D)

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)

        a = dot(grad(u), grad(v)) * dx
        L = self.f * v * dx - self.g * v * ds

        # Compute solution
        u = Function(V)
        solve(a == L, u, bc_D)

        # plot(u)
        # plot(mesh)

        self.u = u

    def save_u(self):
        heat_map = [0 for _ in range(self.grids ** 2)]
        UU = np.zeros(1)
        x = np.linspace(0, self.length, self.grids)
        y = np.linspace(0, self.length, self.grids)
        XX, YY = np.meshgrid(x, y)
        XY = np.hstack((XX.reshape(-1, 1), YY.reshape(-1, 1)))

        for i, xy in enumerate(XY):
            self.u.eval(UU, xy)
            heat_map[i] = UU[0]

        heat_map = np.array(heat_map)
        heat_map = heat_map.reshape(self.grids, self.grids).T

        scio.savemat(self.save_mat, {'list': [self.lst + 1], 'u': heat_map})

    def gen_and_save(self, lst, save_mat="1.mat"):
        self.lst = np.array(lst) - 1
        self.save_mat = save_mat
        self.gen_u()
        self.save_u()


if __name__ == "__main__":
    mg = MapGenerator()
    mg.gen_and_save([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], save_mat="11.mat")
