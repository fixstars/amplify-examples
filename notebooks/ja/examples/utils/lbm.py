# A class to solve a 2D flow field based on a lattice-Boltzmann method (2D9Q).

# YM: Classified, 24th March, 2023
# YM: Computes some statistics and forces, 23rd March, 2023

# Original copyright statements....
# Copyright 2013, Daniel V. Schroeder (Weber State University) 2013

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated data and documentation (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# Except as contained in this notice, the name of the author shall not be used in
# advertising or otherwise to promote the sale, use or other dealings in this
# Software without prior written authorization.

# Credits:
# The "wind tunnel" entry/exit conditions are inspired by Graham Pullan's code
# (http://www.many-core.group.cam.ac.uk/projects/LBdemo.shtml).  Additional inspiration from
# Thomas Pohl's applet (http://thomas-pohl.info/work/lba.html).  Other portions of code are based
# on Wagner (http://www.ndsu.edu/physics/people/faculty/wagner/lattice_boltzmann_codes/) and
# Gonsalves (http://www.physics.buffalo.edu/phy411-506-2004/index.html; code adapted from Succi,
# http://global.oup.com/academic/product/the-lattice-boltzmann-equation-9780199679249).

# For related materials see http://physics.weber.edu/schroeder/fluids

import time
import matplotlib.pyplot as plt
import numpy as np


class Solver:
    def __init__(self, nx=200, ny=80, u=0.1, vis=0.02):
        # Define constants:
        self.barr_x, self.barr_y, self.barr_count = 0, 0, 0
        self.nx, self.ny = nx, ny  # lattice dimensions
        self.tau = 3 * vis + 0.5  # relaxation time, unity dt is assumed.
        self.omega = 1 / self.tau  # relaxation parameter
        self.u_inflow = u  # initial and in-flow speed
        self.four9ths = 4.0 / 9.0  # lattice-Boltzmann weight factors
        self.one9th = 1.0 / 9.0
        self.one36th = 1.0 / 36.0
        self.time = 0
        self.start_time = time.perf_counter()
        self.fx, self.fy = 0, 0
        self.fx_average, self.fy_average, self.average_count = 0, 0, 0
        self.stat_start_time = int(0.5 * self.nx / self.u_inflow)  # flow-through times
        # Initialize all the arrays to steady rightward flow:
        self.second = 3 * self.u_inflow
        self.third = 4.5 * self.u_inflow**2
        self.fourth = -1.5 * self.u_inflow**2
        self.n0 = self.four9ths * (
            np.ones((self.ny, self.nx)) + self.fourth
        )  # particle densities along 9 directions
        self.nN = self.one9th * (np.ones((self.ny, self.nx)) + self.fourth)
        self.nS = self.one9th * (np.ones((self.ny, self.nx)) + self.fourth)
        self.nE = np.zeros((self.ny, self.nx))
        self.nW = np.zeros((self.ny, self.nx))
        self.nNE = np.zeros((self.ny, self.nx))
        self.nSE = np.zeros((self.ny, self.nx))
        self.nNW = np.zeros((self.ny, self.nx))
        self.nSW = np.zeros((self.ny, self.nx))
        self.__set_inflow(key="ic")
        self.rho = None
        self.ux = None
        self.uy = None
        self.__comput_u_rho()
        self.barrier = np.zeros((self.ny, self.nx), bool)  # True at barrier

    def add_model(self, model, ipos=-1, jpos=-1):
        if ipos < 0 and jpos < 0:
            ipos = int(self.nx / 3)
            jpos = int(self.ny / 2)
        for i in range(len(model.xp_int)):
            if model.xp_int[i] + ipos < 0 or model.xp_int[i] + ipos > self.nx - 1:
                continue
            if model.yp_int[i] + jpos < 0 or model.yp_int[i] + jpos > self.ny - 1:
                continue
            self.barrier[model.yp_int[i] + jpos, model.xp_int[i] + ipos] = True
            self.barr_x += model.xp_int[i] + ipos
            self.barr_y += model.yp_int[i] + jpos
            self.barr_count += 1

    def __set_initial(self):
        # Initialize physical boundary:
        self.barrierN = np.roll(self.barrier, 1, axis=0)  # sites just north of barriers
        self.barrierS = np.roll(
            self.barrier, -1, axis=0
        )  # sites just south of barriers
        self.barrierE = np.roll(self.barrier, 1, axis=1)
        self.barrierW = np.roll(self.barrier, -1, axis=1)
        self.barrierNE = np.roll(self.barrierN, 1, axis=1)
        self.barrierNW = np.roll(self.barrierN, -1, axis=1)
        self.barrierSE = np.roll(self.barrierS, 1, axis=1)
        self.barrierSW = np.roll(self.barrierS, -1, axis=1)
        # Centre position (well, sort of) of physical boundaries
        if self.barr_count != 0:
            self.barr_x = int(self.barr_x / self.barr_count)
            self.barr_y = int(self.barr_y / self.barr_count)

    def __comput_u_rho(self):
        self.rho = (
            self.n0
            + self.nN
            + self.nS
            + self.nE
            + self.nW
            + self.nNE
            + self.nSE
            + self.nNW
            + self.nSW
        )
        self.ux = (
            self.nE + self.nNE + self.nSE - self.nW - self.nNW - self.nSW
        ) / self.rho
        self.uy = (
            self.nN + self.nNE + self.nNW - self.nS - self.nSE - self.nSW
        ) / self.rho

    def __set_inflow(self, key="bc"):
        iend = 1
        if key == "ic":
            iend = self.nx
        # Force steady rightward flow at ends (no need to set 0, N, and S components):
        east = 1 + self.second + self.third + self.fourth
        west = 1 - self.second + self.third + self.fourth
        self.nE[:, 0:iend] = self.one9th * east
        self.nW[:, 0:iend] = self.one9th * west
        self.nNE[:, 0:iend] = self.one36th * east
        self.nSE[:, 0:iend] = self.one36th * east
        self.nNW[:, 0:iend] = self.one36th * west
        self.nSW[:, 0:iend] = self.one36th * west
        if key == "bc":
            for j in [0, self.ny - 1]:
                self.nE[j, 0 : self.nx] = self.one9th * east
                self.nW[j, 0 : self.nx] = self.one9th * west
                self.nNE[j, 0 : self.nx] = self.one36th * east
                self.nSE[j, 0 : self.nx] = self.one36th * east
                self.nNW[j, 0 : self.nx] = self.one36th * west
                self.nSW[j, 0 : self.nx] = self.one36th * west

    def __step(self):
        # Move all particles by one step along their directions of motion:
        self.nN = np.roll(
            self.nN, 1, axis=0
        )  # axis 0 is north-south; + direction is north
        self.nNE = np.roll(self.nNE, 1, axis=0)
        self.nNW = np.roll(self.nNW, 1, axis=0)
        self.nS = np.roll(self.nS, -1, axis=0)
        self.nSE = np.roll(self.nSE, -1, axis=0)
        self.nSW = np.roll(self.nSW, -1, axis=0)
        self.nE = np.roll(
            self.nE, 1, axis=1
        )  # axis 1 is east-west; + direction is east
        self.nNE = np.roll(self.nNE, 1, axis=1)
        self.nSE = np.roll(self.nSE, 1, axis=1)
        self.nW = np.roll(self.nW, -1, axis=1)
        self.nNW = np.roll(self.nNW, -1, axis=1)
        self.nSW = np.roll(self.nSW, -1, axis=1)
        # Use boolean arrays to handle barrier collisions (bounce-back):
        self.nN[self.barrierN] = self.nS[self.barrier]
        self.nS[self.barrierS] = self.nN[self.barrier]
        self.nE[self.barrierE] = self.nW[self.barrier]
        self.nW[self.barrierW] = self.nE[self.barrier]
        self.nNE[self.barrierNE] = self.nSW[self.barrier]
        self.nNW[self.barrierNW] = self.nSE[self.barrier]
        self.nSE[self.barrierSE] = self.nNW[self.barrier]
        self.nSW[self.barrierSW] = self.nNE[self.barrier]
        # Collide particles within each cell to redistribute velocities (could be optimized a little more):
        self.__comput_u_rho()
        squx = self.ux * self.ux
        squy = self.uy * self.uy
        u2 = squx + squy
        omu215 = 1 - 1.5 * u2
        uxuy2 = (self.ux * self.uy) * 2
        omgrho = self.omega * self.rho
        one_omg = 1 - self.omega
        self.n0 = one_omg * self.n0 + omgrho * self.four9ths * omu215
        self.nN = one_omg * self.nN + omgrho * self.one9th * (
            omu215 + 3 * self.uy + 4.5 * squy
        )
        self.nS = one_omg * self.nS + omgrho * self.one9th * (
            omu215 - 3 * self.uy + 4.5 * squy
        )
        self.nE = one_omg * self.nE + omgrho * self.one9th * (
            omu215 + 3 * self.ux + 4.5 * squx
        )
        self.nW = one_omg * self.nW + omgrho * self.one9th * (
            omu215 - 3 * self.ux + 4.5 * squx
        )
        self.nNE = one_omg * self.nNE + omgrho * self.one36th * (
            omu215 + 3 * (self.ux + self.uy) + 4.5 * (u2 + uxuy2)
        )
        self.nNW = one_omg * self.nNW + omgrho * self.one36th * (
            omu215 + 3 * (-self.ux + self.uy) + 4.5 * (u2 - uxuy2)
        )
        self.nSE = one_omg * self.nSE + omgrho * self.one36th * (
            omu215 + 3 * (self.ux - self.uy) + 4.5 * (u2 - uxuy2)
        )
        self.nSW = one_omg * self.nSW + omgrho * self.one36th * (
            omu215 + 3 * (-self.ux - self.uy) + 4.5 * (u2 + uxuy2)
        )
        self.__set_inflow()
        self.__get_force()
        self.time += 1

    # Compute curl of the macroscopic velocity field:
    def curl(self):
        return (
            np.roll(self.uy, -1, axis=1)
            - np.roll(self.uy, 1, axis=1)
            - np.roll(self.ux, -1, axis=0)
            + np.roll(self.ux, 1, axis=0)
        )

    def __get_force(self):
        if self.time % 100 != 0:
            return
        self.fx = np.sum(
            self.nE[self.barrier]
            + self.nNE[self.barrier]
            + self.nSE[self.barrier]
            - self.nW[self.barrier]
            - self.nNW[self.barrier]
            - self.nSW[self.barrier]
        )
        self.fy = np.sum(
            self.nN[self.barrier]
            + self.nNE[self.barrier]
            + self.nNW[self.barrier]
            - self.nS[self.barrier]
            - self.nSE[self.barrier]
            - self.nSW[self.barrier]
        )
        if self.time < self.stat_start_time:
            return
        self.average_count += 1
        self.fx_average = self.__welford_online(
            self.fx_average, self.fx, self.average_count
        )
        self.fy_average = self.__welford_online(
            self.fy_average, self.fy, self.average_count
        )

    def __welford_online(self, x_bar_prev, x, count):
        return x_bar_prev + (x - x_bar_prev) / count

    def draw(self):
        plt.clf()
        x, y = np.meshgrid([i for i in range(self.nx)], [i for i in range(self.ny)])
        plt.contourf(
            x, y, self.curl(), cmap="jet", norm=plt.Normalize(-0.1, 0.1), levels=64
        )
        plt.axis("scaled")
        plt.contour(x, y, self.barrier, linewidths=1, colors="k")
        mes = f"time:{self.time}, fx_average:{self.fx_average:.2f}, fy_average:{self.fy_average:.2f}, CPU time:{time.perf_counter() - self.start_time:.1f}s"
        plt.annotate(mes, xy=(5, self.ny - 10), size=10, color="k")
        nskp = int(self.ny / 10)
        ista = int(nskp / 2) - 1
        x, y = np.meshgrid(
            [i for i in range(ista, self.nx, nskp)],
            [i for i in range(ista, self.ny, nskp)],
        )
        u, v = (
            self.ux[ista : self.ny : nskp, ista : self.nx : nskp],
            self.uy[ista : self.ny : nskp, ista : self.nx : nskp],
        )
        plt.quiver(x, y, u, v, scale=5, width=0.002)
        if self.time > self.stat_start_time:
            plt.arrow(
                x=self.barr_x,
                y=self.barr_y,
                dx=self.fx_average * 40,
                dy=self.fy_average * 40,
                head_width=6,
                color="r",
            )
        plt.pause(0.001)

    def __is_diverge(self):
        if np.isnan(np.max(self.ux)) or np.isnan(np.max(self.uy)):
            self.fx = 1e-10
            self.fy = 0
            self.fx_average = 1e-10
            self.fy_average = 0

    def integrate(self, steps=20):
        self.__set_initial()
        for step in range(steps):
            self.__step()
        self.__is_diverge()
        self.draw()
        return self.fx_average, self.fy_average
