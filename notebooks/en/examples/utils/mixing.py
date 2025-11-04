import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh


class MixingSimulator:
    """コンストラクタに与えられたパラメータに基づく frozen flow field を生成し、
    流体の移流拡散シミュレーションを行うクラス。
    """

    def __init__(
        self,
        n_blades: int = 10,
        blade_length: float = 10,
        offset_angle_deg: float = 0,
        eps_radius: float = 10,
        n_layers=2,
        grid_size: int = 50,
    ):
        self._grid_size = grid_size
        self._Y, self._X = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

        # 移流拡散シミュレーションに関係する初期化
        self._diff = 0.1
        self._c: np.ndarray | None = None
        self._std: list[float] = []

        # 流体速度に関する初期化
        self._target_urms = 1.0  # m/s
        self._u: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._set_velocity_field(
            n_blades, blade_length, offset_angle_deg, eps_radius, n_layers
        )

    def _set_velocity_field(
        self,
        n_blades: int = 10,
        blade_length: float = 10,
        offset_angle_deg: float = 0,
        eps_radius: float = 10,
        n_layers=2,
    ) -> None:
        """単位 u_rms を有する frozen flow field を生成する関数。"""
        u_total = np.zeros_like(self._X, dtype=float)
        v_total = np.zeros_like(self._Y, dtype=float)

        for i in range(1, n_layers + 1):
            for j in range(n_blades * i):
                # inner angle between vortex-like structures at each layer
                theta = 2 * np.pi * j / n_blades / i + np.radians(offset_angle_deg)
                # distance to vortex-like structures at each layer
                bx = self._grid_size / 2 + i * blade_length * np.cos(theta)
                by = self._grid_size / 2 + i * blade_length * np.sin(theta)

                dx = self._X - bx
                dy = self._Y - by
                r2 = dx**2 + dy**2

                factor = (-1) ** i * np.exp(-r2 / (2 * eps_radius**2))

                u_total += -factor * dy
                v_total += factor * dx
        urms = np.sqrt(0.5 * (u_total.std() ** 2 + v_total.std() ** 2))
        self._u = u_total * self._target_urms / urms
        self._v = v_total * self._target_urms / urms

    def _initialize_concentration(self) -> np.ndarray:
        x_center = self._grid_size / 2
        y_center = self._grid_size / 4
        radius = self._grid_size / 8
        return (
            np.sqrt((self._X - x_center) ** 2 + (self._Y - y_center) ** 2) < radius
        ).astype(float)

    @property
    def initial_mean(self) -> float:
        return float(self._initialize_concentration().mean())

    def _step(self):
        def apply_neumann_bc(c) -> None:
            c[:, 0] = c[:, 1]
            c[:, -1] = c[:, -2]
            c[0, :] = c[1, :]
            c[-1, :] = c[-2, :]

        assert self._u is not None
        assert self._v is not None
        assert self._c is not None

        dx = dy = 1.0
        dt = 0.3 * min(dx, dy) / max(np.abs(self._u).max(), np.abs(self._v).max())

        # 移流項（1次精度風上）
        adv_x = np.zeros_like(self._c)
        adv_y = np.zeros_like(self._c)
        pos_u = self._u[1:-1, 1:-1] > 0
        neg_u = ~pos_u
        adv_x[1:-1, 1:-1] = pos_u * (
            self._u[1:-1, 1:-1] * (self._c[1:-1, 1:-1] - self._c[0:-2, 1:-1]) / dx
        ) + neg_u * (
            self._u[1:-1, 1:-1] * (self._c[2:, 1:-1] - self._c[1:-1, 1:-1]) / dx
        )
        pos_v = self._v[1:-1, 1:-1] > 0
        neg_v = ~pos_v
        adv_y[1:-1, 1:-1] = pos_v * (
            self._v[1:-1, 1:-1] * (self._c[1:-1, 1:-1] - self._c[1:-1, 0:-2]) / dy
        ) + neg_v * (
            self._v[1:-1, 1:-1] * (self._c[1:-1, 2:] - self._c[1:-1, 1:-1]) / dy
        )

        # 拡散項（2次精度中心）
        diff = self._diff * (
            (self._c[2:, 1:-1] - 2 * self._c[1:-1, 1:-1] + self._c[:-2, 1:-1]) / dx**2
            + (self._c[1:-1, 2:] - 2 * self._c[1:-1, 1:-1] + self._c[1:-1, :-2]) / dy**2
        )

        # 時間発展
        self._c[1:-1, 1:-1] += dt * (-adv_x[1:-1, 1:-1] - adv_y[1:-1, 1:-1] + diff)

        # ノイマン境界条件
        apply_neumann_bc(self._c)

        # 濃度の再初期化
        self._c = np.clip(self._c, 0, 1)
        self._c = self._c * self.initial_mean / self._c.mean()

    def simulate(self, duration: int, num_frames=100) -> float:
        """移流拡散シミュレーションを duration の時間長さ実行し、最終的な濃度の標準偏差を返却する関数。"""
        self._c = self._initialize_concentration()
        assert self._c is not None
        self._c_hist: dict[int, tuple[float, np.ndarray]] = {
            0: (self._c.std(), self._c.copy())
        }

        for ts in range(duration):
            self._step()
            assert self._c is not None
            c_std = self._c.std()
            self._std.append(c_std)
            if (ts + 1) % (duration // num_frames) == 0:
                self._c_hist[ts + 1] = (c_std, self._c.copy())

        return c_std

    def plot_evolution(self, num_snaps=5) -> None:
        """濃度の時間発展をプロットする関数。"""

        def plot(i: int, ax) -> QuadMesh:
            mesh = ax.pcolormesh(
                self._X,
                self._Y,
                list(self._c_hist.values())[i][1],
                cmap="gray_r",
                vmin=0,
                vmax=0.5,
            )
            ax.set_title(f"time: {list(self._c_hist.keys())[i]}", fontsize=8)
            ax.annotate(
                f"std: {list(self._c_hist.values())[i][0]:.3f}",
                xy=(0.05, 0.85),
                xycoords="axes fraction",
                fontsize=8,
            )
            ax.set_aspect("equal")
            ax.set_xlim(0, self._grid_size)
            ax.set_ylim(0, self._grid_size)
            ax.set_xticks([])
            ax.set_yticks([])
            return mesh

        cols = 5
        rows = int(np.ceil(num_snaps / cols))
        _, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))
        axes = np.atleast_1d(axes).flatten()

        n = len(self._c_hist)
        std_min = np.min([v[0] for v in self._c_hist.values()])
        std_max = np.max([v[0] for v in self._c_hist.values()])
        std_delta = (std_max - std_min) / (num_snaps - 1)

        i_snap = 0
        for i in range(n - 1):
            std_left = list(self._c_hist.values())[i + 1][0]
            std_right = list(self._c_hist.values())[i][0]
            std_val = std_min + std_delta * (num_snaps - i_snap - 1)
            if i == 0 or std_left <= std_val <= std_right:
                plot(i, axes[i_snap])
                i_snap += 1
                if i_snap >= num_snaps - 1:
                    break
        plot(n - 1, axes[num_snaps - 1])
        plt.show()
