# YM: c=40 is the default value. 3rd April 2023
# A class to generate an aerofoil profile based on Joukowski transform.
# Original intension is to use this with lbm.Solver class.

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

pd.set_option("display.max_columns", 50)


class Wing:
    def __init__(self, xp, yp):
        self.xp = xp
        self.yp = yp
        self.xp_int = np.array(self.xp, int)
        self.yp_int = np.array(self.yp, int)

    def draw(self):
        plt.plot(self.xp, self.yp, linestyle="-", color="k")
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$y$", fontsize=16)
        plt.tick_params(labelsize=16)
        plt.axis("scaled")
        plt.yticks(
            range(
                (int(np.min(self.yp) / 10) - 1) * 10,
                (int(np.max(self.yp) / 10) + 1) * 10 + 1,
                10,
            )
        )
        plt.xticks(
            range(
                (int(np.min(self.xp) / 10) - 1) * 10,
                (int(np.max(self.xp) / 10) + 1) * 10 + 1,
                20,
            )
        )
        plt.pause(0.001)


class WingGenerator:
    def __init__(
        self,
        c=40.0,
        min_xi0=1,
        max_xi0=10,
        nbins_xi0=20,
        min_eta0=0,
        max_eta0=10,
        nbins_eta0=40,
        min_alpha=0,
        max_alpha=40,
        nbins_alpha=40,
        nbins=1000,
    ):
        self.c = c
        # xi is for thickness
        self.min_xi0 = min_xi0
        self.max_xi0 = max_xi0
        self.nbins_xi0 = nbins_xi0
        self.xi0_tab = np.array(
            [i / nbins_xi0 * (max_xi0 - min_xi0) + min_xi0 for i in range(nbins_xi0)]
        )
        # eta is for curvature
        self.min_eta0 = min_eta0
        self.max_eta0 = max_eta0
        self.nbins_eta0 = nbins_eta0
        self.eta0_tab = np.zeros(nbins_eta0)
        self.eta0_tab = np.array(
            [
                i / nbins_eta0 * (max_eta0 - min_eta0) + min_eta0
                for i in range(nbins_eta0)
            ]
        )
        # alpha is for attack angle in degree
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.nbins_alpha = nbins_alpha
        self.alpha_tab = np.zeros(nbins_alpha)
        self.alpha_tab = np.array(
            [
                i / nbins_alpha * (max_alpha - min_alpha) + min_alpha
                for i in range(nbins_alpha)
            ]
        )
        self.xi0 = None
        self.eta0 = None
        self.alpha = None
        self.nbins = nbins
        self.i_xi0 = 0
        self.i_eta0 = 0
        self.i_alpha = 0

    def show_index_table(self):
        print("[Index table]")
        nbins_max = max(self.nbins_xi0, self.nbins_eta0, self.nbins_alpha)
        df = pd.DataFrame(index=[], columns=range(nbins_max))
        df.loc["xi0[i]"] = np.append(
            np.round(self.xi0_tab, decimals=2),
            np.full(nbins_max - self.nbins_xi0, "-"),
        )
        df.loc["eta0[i]"] = np.append(
            np.round(self.eta0_tab, decimals=2),
            np.full(nbins_max - self.nbins_eta0, "-"),
        )
        df.loc["alpha[i]"] = np.append(
            np.round(self.alpha_tab, decimals=2),
            np.full(nbins_max - self.nbins_alpha, "-"),
        )
        print(df)

    def show_parameters(self, name):
        print(
            f"{name}: xi0[{self.i_xi0}]={self.xi0:.2f}, eta0[{self.i_eta0}]={self.eta0:.2f}, alpha[{self.i_alpha}]={self.alpha:.2f}deg"
        )

    def generate_wing(self, i_xi0=0, i_eta0=0, i_alpha=0):
        self.i_xi0 = i_xi0
        self.i_eta0 = i_eta0
        self.i_alpha = i_alpha
        self.xi0 = self.xi0_tab[self.i_xi0]
        self.eta0 = self.eta0_tab[self.i_eta0]
        self.alpha = self.alpha_tab[self.i_alpha]
        R = math.sqrt((self.c - self.xi0) * (self.c - self.xi0) + self.eta0 * self.eta0)
        theta = np.array(
            [i / (self.nbins - 1) * 2 * math.pi for i in range(self.nbins)]
        )
        xi = np.array([R * math.cos(theta[i]) + self.xi0 for i in range(self.nbins)])
        eta = np.array([R * math.sin(theta[i]) + self.eta0 for i in range(self.nbins)])
        x = np.array(
            [
                xi[i] + self.c * self.c * xi[i] / (xi[i] * xi[i] + eta[i] * eta[i])
                for i in range(self.nbins)
            ]
        )
        y = np.array(
            [
                eta[i] - self.c * self.c * eta[i] / (xi[i] * xi[i] + eta[i] * eta[i])
                for i in range(self.nbins)
            ]
        )
        xp = np.array(
            [
                x[i] * math.cos(math.radians(-self.alpha))
                - y[i] * math.sin(math.radians(-self.alpha))
                for i in range(self.nbins)
            ]
        )
        yp = np.array(
            [
                x[i] * math.sin(math.radians(-self.alpha))
                + y[i] * math.cos(math.radians(-self.alpha))
                for i in range(self.nbins)
            ]
        )
        # offset the wing so that it is centered at 0.
        x_shift = int((np.max(xp) + np.min(xp)) / 2)
        y_shift = int((np.max(yp) + np.min(yp)) / 2)
        xp -= x_shift
        yp -= y_shift
        return Wing(xp, yp)
