import random
from bisect import bisect_right
from enum import Enum, IntEnum, auto
from math import sqrt, tanh
from random import randint
from random import seed as set_seed
from typing import Type

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display import display
from matplotlib.animation import FuncAnimation

# モデルパラメータ（定数）
# 長さはm単位
# 時間は秒単位
ROAD_WIDTH = 2.0
CAR_SIZE = 1.8
SIGNAL_SIZE = 2.2
CROSSPOINT_SIZE = 2.0

LIMIT_SPEED = 4.0  # m/s
SENSITIVITY = 1.0
SEC_PER_TICK = 0.1

COLLISION = LIMIT_SPEED * SEC_PER_TICK


# 列挙型
class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SignalColor(Enum):
    RED = auto()
    GREEN = auto()


# 逆の関係
def reverse(direction: Direction):
    if direction == Direction.UP:
        return Direction.DOWN
    elif direction == Direction.DOWN:
        return Direction.UP
    elif direction == Direction.LEFT:
        return Direction.RIGHT
    elif direction == Direction.RIGHT:
        return Direction.LEFT
    else:
        print("ERROR:direction is wrong")


# 左の関係
def left(direction: Direction):
    if direction == Direction.UP:
        return Direction.LEFT
    elif direction == Direction.DOWN:
        return Direction.RIGHT
    elif direction == Direction.LEFT:
        return Direction.DOWN
    elif direction == Direction.RIGHT:
        return Direction.UP
    else:
        print("ERROR:direction is wrong")


# 右の関係
def right(direction: Direction):
    return reverse(left(direction))


# 方向単位ベクトルの定義
def dir_to_vec(direction: Direction):
    if direction == Direction.UP:
        return np.array([0, 1])
    elif direction == Direction.DOWN:
        return np.array([0, -1])
    elif direction == Direction.LEFT:
        return np.array([-1, 0])
    elif direction == Direction.RIGHT:
        return np.array([1, 0])
    else:
        print("ERROR:direction is wrong")


# 車オブジェクト
class Car:
    def __init__(
        self,
        x: float,
        y: float,
        velocity: float,
        direction: Direction,
        map_size: int,
        limit_speed: float,
        color="#ffffff",
        round_trip_wait_time=-1,
    ) -> None:
        self.x = x
        self.y = y
        self.velocity = velocity
        self.direction = direction
        self.map_size = map_size
        self.limit_speed = limit_speed
        self.car_size = CAR_SIZE
        self.startpoint = None
        self.goalpoint = None
        self.draw_sgpoint = True
        self.color = color
        self.round_trip_wait_time = round_trip_wait_time  # 往復時の待ち時間(-1:往復しない)
        self.wait_count = 0
        # self.from_index = None # どの信号から走り始めたのか

    def set_startpoint(self, x: float, y: float, direction: Direction):
        self.startpoint = np.array([x, y])
        self.sdirection = direction

    def set_goalpoint(self, x: float, y: float, direction: Direction):
        self.goalpoint = np.array([x, y])
        self.gdirection = direction

    # 現在位置から目的地に向かうためのルートを決定。
    # 交差点でどちらの方向に行くのか。
    def decide_direction(self, choices: list):
        # ゴール地点が設定されていなければ直進を選択
        if self.goalpoint is None:
            return 0
        wanted_vec = self.goalpoint - np.array([self.x, self.y])
        wanted_vec /= np.linalg.norm(wanted_vec)
        opt_index = None
        opt_ip = -1.0

        for i in range(len(choices)):
            ip = wanted_vec.dot(choices[i])
            if opt_ip < ip:
                opt_ip = ip
                opt_index = i

        return opt_index

    # 目的地に到達したら削除依頼をStreetにする
    def is_arrived(self):
        if self.goalpoint is not None:
            epsilon = sqrt(COLLISION**2 + ROAD_WIDTH**2)
            if (self.x - self.goalpoint[0]) ** 2 + (
                self.y - self.goalpoint[1]
            ) ** 2 < epsilon**2:
                return True
            else:
                return False

    def draw(self, ax):
        ax.add_patch(
            patches.Rectangle(
                xy=(self.x - self.car_size / 2, self.y - self.car_size / 2),
                width=self.car_size,
                height=self.car_size,
                fc=self.color,
                ec="#000000",
                fill=True,
            )
        )
        # 開始地点と目的地点の表示
        if self.draw_sgpoint and self.goalpoint is not None:
            offset = ROAD_WIDTH * 2
            if self.gdirection == Direction.UP:
                offset = np.array([-offset, 0])
            elif self.gdirection == Direction.DOWN:
                offset = np.array([offset, 0])
            elif self.gdirection == Direction.LEFT:
                offset = np.array([0, -offset])
            elif self.gdirection == Direction.RIGHT:
                offset = np.array([0, offset])
            else:
                print("ERROR:Direction is wrong")
            gxy = self.goalpoint + offset
            if self.gdirection == Direction.DOWN:
                gxy = (gxy[0] % self.map_size, gxy[1])
            elif self.gdirection == Direction.LEFT:
                gxy = (gxy[0], gxy[1] % self.map_size)

            ax.add_patch(
                patches.Circle(
                    xy=gxy,
                    radius=ROAD_WIDTH / 2,
                    # fc="#ffffff",
                    fc=self.color,
                    ec="#000000",
                    fill=True,
                    alpha=1.0,
                )
            )

    def forward(self):
        if self.direction == Direction.UP:
            self.y += self.velocity * SEC_PER_TICK
        elif self.direction == Direction.DOWN:
            self.y -= self.velocity * SEC_PER_TICK
        elif self.direction == Direction.RIGHT:
            self.x += self.velocity * SEC_PER_TICK
        elif self.direction == Direction.LEFT:
            self.x -= self.velocity * SEC_PER_TICK
        else:
            print("ERROR1: Car direction is an exception value.")
            print(self.direction)

        self.x = (self.x + self.map_size) % self.map_size
        self.y = (self.y + self.map_size) % self.map_size

    def update_velocity(self, forward_car: "Car"):
        a = SENSITIVITY  # 感応度(反射神経のようなもの)
        if forward_car == self:
            # self.velocity = self.limit_speed
            self.velocity += a * (self.limit_speed - self.velocity)
        else:
            if self.direction == Direction.UP:
                dist = forward_car.y - self.y
            elif self.direction == Direction.DOWN:
                dist = self.y - forward_car.y
            elif self.direction == Direction.RIGHT:
                dist = forward_car.x - self.x
            elif self.direction == Direction.LEFT:
                dist = self.x - forward_car.x
            else:
                print("ERROR2: Car direction is an exception value.")

            # 画面端の処理
            if dist < 0:
                dist += self.map_size
            # 車体がぶつからない車間距離
            dist -= self.car_size

            # self.velocity = self.optimal_speed(dist)
            self.velocity += a * (self.optimal_speed(dist) - self.velocity)

    # 車間距離から計算される最適速度を定義
    # 制限速度は超えないように
    def optimal_speed(self, x):
        speed = self.limit_speed * (tanh(x - 2) + tanh(2)) / 2  # Realistic model
        return speed if speed >= 0 else 0
        # return self.limit_speed * tanh(x)*2 # simple model


# 信号オブジェクト
class Signal:
    def __init__(
        self,
        x: float,
        y: float,
        direction: Direction,
        cycle_span: int,
        color: SignalColor,
    ) -> None:
        self.x = x
        self.y = y
        self.direction = direction
        self.signal_size = SIGNAL_SIZE
        self.cycle_span = cycle_span
        self.color = color
        self.life = 0
        if direction == Direction.UP or direction == Direction.DOWN:
            self.pos = self.y
        else:
            self.pos = self.x

    def draw(self, ax):
        if self.color == SignalColor.RED:
            fillcolor = "red"
        elif self.color == SignalColor.GREEN:
            fillcolor = "green"
        else:
            print("ERROR3: Signal Color is an exception value.")
            return

        ax.add_patch(
            patches.Circle(
                xy=(self.x, self.y),
                radius=self.signal_size / 2,
                fc=fillcolor,
                fill=True,
                alpha=0.5,
            )
        )

    def update_state(self):
        self.life += 1
        if self.life >= self.cycle_span:
            self.life -= self.cycle_span

            if self.color == SignalColor.RED:
                self.color = SignalColor.GREEN
            elif self.color == SignalColor.GREEN:
                self.color = SignalColor.RED
            else:
                print("ERROR: Signal Color is an exception value.")
                return

    def stop_car(self, car: "Car"):
        if self.color != SignalColor.RED:
            return

        if self.direction == Direction.UP or self.direction == Direction.DOWN:
            if (
                self.y - self.signal_size / 2 < car.y
                and car.y < self.y + self.signal_size / 2
            ):
                car.velocity = 0.0
        elif self.direction == Direction.LEFT or self.direction == Direction.RIGHT:
            if (
                self.x - self.signal_size / 2 < car.x
                and car.x < self.x + self.signal_size / 2
            ):
                car.velocity = 0.0


# 1D Traffic model
class car_demo:
    def __init__(self) -> None:
        self.map_size = 80
        self.fig, self.ax = plt.subplots()
        self.limit_speed = LIMIT_SPEED
        self.make_carlist()
        self.make_signallist()

    def make_carlist(self):
        self.carlist = []
        for i in range(15):
            self.carlist.append(
                Car(
                    5.0 + i,
                    self.map_size / 2,
                    0.0,
                    Direction.RIGHT,
                    self.map_size,
                    self.limit_speed,
                )
            )

    def make_signallist(self):
        self.signallist = []

    def update_car(self):
        for car in self.carlist:
            car.forward()
            car.draw(self.ax)
        n = len(self.carlist)
        for i in range(n):
            self.carlist[i].update_velocity(self.carlist[(i + 1) % n])

        for signal in self.signallist:
            for car in self.carlist:
                signal.stop_car(car)

    def update_signal(self):
        for signal in self.signallist:
            signal.update_state()
            signal.draw(self.ax)

    def update(self, i):
        self.ax.clear()
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect("equal")
        self.ax.set_title(f"ticks {i}")
        self.update_car()
        self.update_signal()

    def animation(self, interval=100, frames=200, repeat=False):
        return FuncAnimation(
            self.fig, self.update, interval=interval, frames=frames, repeat=repeat
        )


# 各Streetは交差点にこのオブジェクトを持つ。
# leftは左折後のStreet、rightは右折後のStreet
class CrossPoint:
    def __init__(
        self,
        pos: float,
        direction: Direction,
        left: "Street",
        right: "Street",
        map_size,
    ):
        self.pos = pos
        self.direction = direction
        self.point_size = CROSSPOINT_SIZE  # 当たり判定
        self.left = left
        self.right = right
        self.map_size = map_size

    def __str__(self) -> str:
        return f"{(self.pos, self.direction, self.access)}"

    def ask_car(self, car: Car) -> bool:
        if self.direction == Direction.UP or self.direction == Direction.DOWN:
            if (
                self.pos - self.point_size / 2 < car.y
                and car.y < self.pos + self.point_size / 2
            ) or (
                self.pos - self.point_size / 2 < car.y + self.map_size
                and car.y + self.map_size < self.pos + self.point_size / 2
            ):
                return True
        elif self.direction == Direction.LEFT or self.direction == Direction.RIGHT:
            if (
                self.pos - self.point_size / 2 < car.x
                and car.x < self.pos + self.point_size / 2
            ) or (
                self.pos - self.point_size / 2 < car.x + self.map_size
                and car.x + self.map_size < self.pos + self.point_size / 2
            ):
                return True

        return False


# 交差点の信号制御
class ControlSignal:
    def __init__(self, signal_list: list, term1: float, term2: float, phase: float):
        self.signal_list = signal_list
        self.term1 = term1
        self.term2 = term2
        self.time = phase % (term1 + term2)

    def update_state(self):
        self.time = (self.time + SEC_PER_TICK) % (self.term1 + self.term2)
        if self.time < self.term2:
            self.phase1()
        else:
            self.phase2()

    # フェーズ1とフェーズ2の定義
    def phase1(self):
        self.signal_list[Direction.UP].color = SignalColor.RED
        self.signal_list[Direction.DOWN].color = SignalColor.RED
        self.signal_list[Direction.LEFT].color = SignalColor.GREEN
        self.signal_list[Direction.RIGHT].color = SignalColor.GREEN

    def phase2(self):
        self.signal_list[Direction.UP].color = SignalColor.GREEN
        self.signal_list[Direction.DOWN].color = SignalColor.GREEN
        self.signal_list[Direction.LEFT].color = SignalColor.RED
        self.signal_list[Direction.RIGHT].color = SignalColor.RED


class Street:
    def __init__(
        self, pos: float, direction: Direction, index: int, map_size: float
    ) -> None:
        self.pos = pos  # 道の左端(UP,DOWN)か下端(LEFT,RIGHT)の座標
        self.direction = direction
        self.index = index
        self.map_size = map_size
        self.car_list = []
        self.signal_list = []
        self.crosspoint_list = []
        self.left_streetlist = []  # 左折リスト
        self.right_streetlist = []  # 右折リスト
        self.rev_street = None
        self.wait_car_list = []  # 待機車のリスト
        self.color = "#ffffff"

    def draw(self, ax):
        if self.direction == Direction.UP or self.direction == Direction.DOWN:
            ax.add_patch(
                patches.Rectangle(
                    xy=(self.pos, 0),
                    height=self.map_size,
                    width=ROAD_WIDTH,
                    fc=self.color,
                    fill=True,
                )
            )
        elif self.direction == Direction.LEFT or self.direction == Direction.RIGHT:
            ax.add_patch(
                patches.Rectangle(
                    xy=(0, self.pos),
                    height=ROAD_WIDTH,
                    width=self.map_size,
                    fc=self.color,
                    fill=True,
                )
            )
        else:
            print("ERROR: street direction is an exception value.")

    def add_car(self, append_car: Car) -> bool:
        epsilon = append_car.car_size
        for street_car in self.car_list:
            # あまりにも近い場合は追加失敗
            if (street_car.x - append_car.x) ** 2 + (
                street_car.y - append_car.y
            ) ** 2 < epsilon**2:
                return False

        append_car.direction = self.direction

        # 2分探索して挿入(の予定だった)
        if self.direction == Direction.UP or self.direction == Direction.DOWN:
            append_car.x = self.pos + ROAD_WIDTH / 2
            # bisect_right(self.car_list, append_car, key=lambda obj: obj.y)
            self.car_list.append(append_car)
            if self.direction == Direction.UP:
                self.car_list.sort(key=lambda obj: obj.y)
            else:
                self.car_list.sort(key=lambda obj: -obj.y)
        elif self.direction == Direction.LEFT or self.direction == Direction.RIGHT:
            append_car.y = self.pos + ROAD_WIDTH / 2

            # bisect_right(self.car_list, append_car, key=lambda obj: obj.x)
            self.car_list.append(append_car)
            self.car_list.sort(key=lambda obj: obj.x)
            if self.direction == Direction.RIGHT:
                self.car_list.sort(key=lambda obj: obj.x)
            else:
                self.car_list.sort(key=lambda obj: -obj.x)
        else:
            print("ERROR: street direction is an exception value.")
            return False

        return True

    def delete_car(self, delete_car: Car):
        # delete_carと同一のものを削除
        self.car_list.remove(delete_car)

    def add_signal(self, signal: Signal):
        self.signal_list.append(signal)

    def add_crosspoint(self, crosspoint: CrossPoint):
        self.crosspoint_list.append(crosspoint)

    def add_left_streetlist(self, street: Type["Street"]):
        self.left_streetlist.append(street)

    def add_right_streetlist(self, street: Type["Street"]):
        self.right_streetlist.append(street)

    def add_wait_car(self, car: Car):
        goal = car.startpoint
        car.startpoint = car.goalpoint
        car.goalpoint = goal
        goal_d = car.sdirection
        car.sdirection = car.gdirection
        car.gdirection = goal_d
        car.direction = reverse(car.direction)
        car.wait_count = 0
        self.rev_street.wait_car_list.append(car)

    def update_car(self):
        # 車の位置と速度の更新
        for car in self.car_list:
            car.forward()
        n = len(self.car_list)
        for i in range(n):
            self.car_list[i].update_velocity(self.car_list[(i + 1) % n])

        # 信号による車の停止(高速化の余地)
        for signal in self.signal_list:
            for car in self.car_list:
                signal.stop_car(car)

        # 待機車の更新
        for car in self.wait_car_list:
            if car.wait_count >= car.round_trip_wait_time:
                car.velocity = 0.1  # 出発速度の初期化
                self.add_car(car)
                self.wait_car_list.remove(car)
            else:
                car.wait_count += SEC_PER_TICK

        # 交差点(信号)での右左折処理（高速化の余地）
        for crosspoint in self.crosspoint_list:
            for car in self.car_list:
                if crosspoint.ask_car(car):
                    res_index = car.decide_direction(
                        list(
                            map(
                                dir_to_vec,
                                [
                                    self.direction,
                                    left(self.direction),
                                    right(self.direction),
                                ],
                            )
                        )
                    )

                    if res_index == 0:
                        pass
                    elif res_index == 1:
                        # 左折操作
                        res = crosspoint.left.add_car(car)
                        if res:
                            # 追加に成功したら車を自分のリストから削除
                            self.delete_car(car)
                            car.velocity = 0.1  # 右左折の徐行
                        else:
                            # 追加に失敗したら車を待機させる。
                            car.velocity = 0.0

                    elif res_index == 2:
                        # 右折操作
                        res = crosspoint.right.add_car(car)
                        if res:
                            # 追加に成功したら車をリストから削除
                            self.delete_car(car)
                            car.velocity = 0.1  # 右左折の徐行
                        else:
                            # 追加に失敗したら車を待機させる。
                            car.velocity = 0.0
                    else:
                        print("ERROR: decide_direction is wrong answer.")

        # 目的地に到達した車の削除
        for car in self.car_list:
            if car.is_arrived():
                if car.round_trip_wait_time != -1:
                    self.add_wait_car(car)
                self.delete_car(car)

    # 車が次の信号に到達したか
    def is_reached_car(self, car: Car) -> bool:
        if car.from_index is None:
            print("ERROR: from_index is None")
            return False

    def update_signal(self):
        for signal in self.signal_list:
            signal.update_state()

    def draw_car(self, ax):
        for car in self.car_list:
            car.draw(ax)

    def draw_signal(self, ax):
        for signal in self.signal_list:
            signal.draw(ax)


# 正方形で実装
class grid_model:
    def __init__(self, n, map_size=80, seed=0) -> None:
        self.map_size = map_size
        self.n = n
        # self.fig, self.ax = plt.subplots()
        self.one_block_size = self.map_size / n - 1
        self.limit_speed = LIMIT_SPEED
        self.street_list = []
        self.make_street()
        self.random_seed = seed
        set_seed(self.random_seed)
        self.stat_velocity = []  # 各tick時点の各車の速度[km/h]
        self.stat_velocity_mean = []  # 各tick時点の各車の速度平均
        self.stat_velocity_std = []  # 各tick時点の各車の速度分散(標準偏差)
        self.tick = 0  # 現在のtick
        self.stat_seq_mean = []

    def make_street(self):
        # street listの作成
        for i in range(4):
            self.street_list.append([])
        for i in range(self.n):
            self.street_list[Direction.LEFT].append(
                Street(i * self.map_size / self.n, Direction.LEFT, i, self.map_size)
            )
            self.street_list[Direction.RIGHT].append(
                Street(
                    i * self.map_size / self.n + ROAD_WIDTH,
                    Direction.RIGHT,
                    i,
                    self.map_size,
                )
            )
        for i in range(self.n):
            self.street_list[Direction.UP].append(
                Street(
                    (i + 1) * self.map_size / self.n - (ROAD_WIDTH * 2),
                    Direction.UP,
                    i,
                    self.map_size,
                )
            )
            self.street_list[Direction.DOWN].append(
                Street(
                    (i + 1) * self.map_size / self.n - ROAD_WIDTH,
                    Direction.DOWN,
                    i,
                    self.map_size,
                )
            )

        # Signalの配置
        for i in range(self.n):  # i:縦
            for j in range(self.n):  # j:横
                # 信号設置のための交差点位置の基準
                basis_y = self.street_list[Direction.LEFT][j].pos
                basis_x = self.street_list[Direction.UP][i].pos
                m = self.map_size

                # 縦方向の信号設置
                pos_UP = (basis_x + ROAD_WIDTH / 2, basis_y - ROAD_WIDTH / 2)
                pos_DOWN = (
                    basis_x + ROAD_WIDTH + ROAD_WIDTH / 2,
                    basis_y + ROAD_WIDTH * 2 + ROAD_WIDTH / 2,
                )
                pos_UP = ((pos_UP[0] + m) % m, (pos_UP[1] + m) % m)
                pos_DOWN = ((pos_DOWN[0] + m) % m, (pos_DOWN[1] + m) % m)
                self.street_list[Direction.UP][i].add_signal(
                    Signal(pos_UP[0], pos_UP[1], Direction.UP, 20, SignalColor.GREEN)
                )
                self.street_list[Direction.DOWN][i].add_signal(
                    Signal(
                        pos_DOWN[0], pos_DOWN[1], Direction.DOWN, 20, SignalColor.GREEN
                    )
                )

                # 横方向の信号設置
                pos_LEFT = (
                    basis_x + ROAD_WIDTH * 2 + ROAD_WIDTH / 2,
                    basis_y + ROAD_WIDTH / 2,
                )
                pos_RIGHT = (
                    basis_x - ROAD_WIDTH / 2,
                    basis_y + ROAD_WIDTH + ROAD_WIDTH / 2,
                )
                pos_LEFT = ((pos_LEFT[0] + m) % m, (pos_LEFT[1] + m) % m)
                pos_RIGHT = ((pos_RIGHT[0] + m) % m, (pos_RIGHT[1] + m) % m)
                self.street_list[Direction.LEFT][j].add_signal(
                    Signal(
                        pos_LEFT[0], pos_LEFT[1], Direction.LEFT, 20, SignalColor.RED
                    )
                )
                self.street_list[Direction.RIGHT][j].add_signal(
                    Signal(
                        pos_RIGHT[0], pos_RIGHT[1], Direction.RIGHT, 20, SignalColor.RED
                    )
                )

        # 交差点の追加
        for i in range(self.n):
            for j, signal in enumerate(self.street_list[Direction.UP][i].signal_list):
                self.street_list[Direction.UP][i].add_crosspoint(
                    CrossPoint(
                        signal.pos + ROAD_WIDTH,
                        signal.direction,
                        self.street_list[Direction.LEFT][j],
                        self.street_list[Direction.RIGHT][j],
                        self.map_size,
                    )
                )
            for j, signal in enumerate(self.street_list[Direction.DOWN][i].signal_list):
                self.street_list[Direction.DOWN][i].add_crosspoint(
                    CrossPoint(
                        signal.pos - ROAD_WIDTH,
                        signal.direction,
                        self.street_list[Direction.RIGHT][j],
                        self.street_list[Direction.LEFT][j],
                        self.map_size,
                    )
                )
            for j, signal in enumerate(
                self.street_list[Direction.RIGHT][i].signal_list
            ):
                self.street_list[Direction.RIGHT][i].add_crosspoint(
                    CrossPoint(
                        signal.pos + ROAD_WIDTH,
                        signal.direction,
                        self.street_list[Direction.UP][j],
                        self.street_list[Direction.DOWN][j],
                        self.map_size,
                    )
                )
            for j, signal in enumerate(self.street_list[Direction.LEFT][i].signal_list):
                self.street_list[Direction.LEFT][i].add_crosspoint(
                    CrossPoint(
                        signal.pos - ROAD_WIDTH,
                        signal.direction,
                        self.street_list[Direction.DOWN][j],
                        self.street_list[Direction.UP][j],
                        self.map_size,
                    )
                )

        # 交差点で繋がった隣のStreetの追加
        for i in range(self.n):
            for j in range(self.n):
                # 右方向のStreetに対する隣接ノードの追加
                self.street_list[Direction.RIGHT][i].add_left_streetlist(
                    self.street_list[Direction.UP][j]
                )
                self.street_list[Direction.RIGHT][i].add_right_streetlist(
                    self.street_list[Direction.DOWN][j]
                )

                # 左方向のStreetに対する隣接ノードの追加
                self.street_list[Direction.LEFT][i].add_left_streetlist(
                    self.street_list[Direction.DOWN][self.n - 1 - j]
                )
                self.street_list[Direction.LEFT][i].add_right_streetlist(
                    self.street_list[Direction.UP][self.n - 1 - j]
                )

                # 上方向のStreetに対する隣接ノードの追加
                self.street_list[Direction.UP][i].add_left_streetlist(
                    self.street_list[Direction.LEFT][j]
                )
                self.street_list[Direction.UP][i].add_right_streetlist(
                    self.street_list[Direction.RIGHT][j]
                )

                # 下方向のStreetに対する隣接ノードの追加
                self.street_list[Direction.DOWN][i].add_left_streetlist(
                    self.street_list[Direction.RIGHT][self.n - 1 - j]
                )
                self.street_list[Direction.DOWN][i].add_right_streetlist(
                    self.street_list[Direction.LEFT][self.n - 1 - j]
                )

        # 逆方向のStreetの参照
        for i in range(self.n):
            for j in range(4):
                self.street_list[j][i].rev_street = self.street_list[reverse(j)][i]

    # 車の個数を指定してランダムに追加
    def make_car(self, car_num: int):
        while car_num:
            for streets in self.street_list:
                for street in streets:
                    if random() > 0.5:
                        if street.add_car(
                            Car(
                                random() * self.map_size,
                                random() * self.map_size,
                                0.1,
                                street.direction,
                                self.map_size,
                                self.limit_speed,
                            )
                        ):
                            car_num -= 1
                        if car_num <= 0:
                            return

    # 特定の道に車を追加
    def random_make_car_on_street(self, direction: Direction, index: int) -> bool:
        return self.street_list[direction][index].add_car(
            Car(
                random() * self.map_size,
                random() * self.map_size,
                0.1,
                direction,
                self.map_size,
                self.limit_speed,
            )
        )

    # 目的地を持った車をランダムにn個配置
    def random_make_car_for_path(self, n, round_trip=False):
        count = n
        while count:
            res = self.make_car_for_path(
                randint(0, 3),
                randint(0, self.n - 1),
                random.random() * self.map_size,
                randint(0, 3),
                randint(0, self.n - 1),
                random.random() * self.map_size,
                round_trip=round_trip,
            )
            if res:
                count -= 1

    # 出発地点と目的地点を指定して車を配置
    def make_car_for_path(
        self,
        start_direction: Direction,
        start_index: int,
        start_pos: float,
        goal_direction: Direction,
        goal_index: int,
        goal_pos: float,
        round_trip=False,
    ):
        # 始点か終点が交差点に位置していたら、車配置失敗(False)
        additional_offset = ROAD_WIDTH / 2
        if start_direction == Direction.UP or start_direction == Direction.DOWN:
            for i in range(self.n):
                if (
                    i * self.map_size / self.n - additional_offset <= start_pos
                    and start_pos
                    <= i * self.map_size / self.n + (ROAD_WIDTH * 2) + additional_offset
                ):
                    return False
        else:
            for i in range(self.n):
                if (i + 1) * self.map_size / self.n - (
                    ROAD_WIDTH * 2
                ) - additional_offset <= start_pos and start_pos <= (
                    i + 1
                ) * self.map_size / self.n + additional_offset:
                    return False
        if goal_direction == Direction.UP or goal_direction == Direction.DOWN:
            for i in range(self.n):
                if (
                    i * self.map_size / self.n - additional_offset <= goal_pos
                    and goal_pos
                    <= i * self.map_size / self.n + (ROAD_WIDTH * 2) + additional_offset
                ):
                    return False
        else:
            for i in range(self.n):
                if (i + 1) * self.map_size / self.n - (
                    ROAD_WIDTH * 2
                ) - additional_offset <= goal_pos and goal_pos <= (
                    i + 1
                ) * self.map_size / self.n + additional_offset:
                    return False

        s_coor = self.index_to_coordinate(start_direction, start_index, start_pos)
        g_coor = self.index_to_coordinate(goal_direction, goal_index, goal_pos)

        # はじめに進むべき向きを決める
        if start_direction == Direction.UP or start_direction == Direction.DOWN:
            if s_coor[1] < g_coor[1]:
                start_direction = Direction.UP
            else:
                start_direction = Direction.DOWN
        elif start_direction == Direction.LEFT or start_direction == Direction.RIGHT:
            if s_coor[0] < g_coor[0]:
                start_direction = Direction.RIGHT
            else:
                start_direction = Direction.LEFT
        else:
            print("ERROR: direction is wrong.")

        # Carのインスタンスを設定
        new_car = Car(
            s_coor[0], s_coor[1], 0.1, start_direction, self.map_size, self.limit_speed
        )
        new_car.set_startpoint(s_coor[0], s_coor[1], start_direction)
        new_car.set_goalpoint(g_coor[0], g_coor[1], goal_direction)
        if round_trip:
            new_car.round_trip_wait_time = randint(0, 20)  # 待機時間の設定

        self.street_list[start_direction][start_index].add_car(new_car)
        return True

    def index_to_coordinate(self, direction: Direction, index, pos):
        if direction == Direction.UP:
            return (self.street_list[direction][index].pos + ROAD_WIDTH, pos)
        elif direction == Direction.DOWN:
            return (self.street_list[direction][index].pos, pos)
        elif direction == Direction.LEFT:
            return (pos, self.street_list[direction][index].pos + ROAD_WIDTH)
        elif direction == Direction.RIGHT:
            return (pos, self.street_list[direction][index].pos)
        else:
            print("ERROR: direction is wrong.")

    def update_car(self):
        for streets in self.street_list:
            for street in streets:
                street.update_car()

    def update_signal(self):
        for streets in self.street_list:
            for street in streets:
                street.update_signal()  # Signalオブジェクトによる色制御

    def update_map(self):
        self.update_car()
        self.update_signal()
        self.tick += 1

    def draw_map(self):
        # 画面の初期化
        self.ax.clear()
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect("equal")
        self.ax.set_title(f"elapsed time {self.tick*SEC_PER_TICK:.3f} s.")

        self.ax.add_patch(
            patches.Rectangle(
                xy=(0, 0),
                height=self.map_size,
                width=self.map_size,
                fc="#cccccc",
                fill=True,
            )
        )
        for streets in self.street_list:
            for street in streets:
                street.draw(self.ax)

        for streets in self.street_list:
            for street in streets:
                street.draw_car(self.ax)
        for streets in self.street_list:
            for street in streets:
                street.draw_signal(self.ax)

    def stat(self):
        # 各車の速度を取得
        tmp = []
        for streets in self.street_list:
            for street in streets:
                for car in street.car_list:
                    tmp.append(car.velocity * 3.6)

        np_tmp = np.array(tmp)
        self.stat_velocity.append(tmp)

        # 統計量
        if np_tmp.shape[0] == 0:
            self.stat_velocity_mean.append(0)
            self.stat_velocity_std.append(0)
        else:
            self.stat_velocity_mean.append(np_tmp.mean())
            self.stat_velocity_std.append(np_tmp.std())

        length = len(self.stat_seq_mean)
        if length == 0:
            self.stat_seq_mean.append(np_tmp.mean())
        else:
            self.stat_seq_mean.append(
                (self.stat_seq_mean[-1] * (self.tick - 1) + np_tmp.mean()) / (self.tick)
            )

    def update(self, i):
        self.update_map()
        self.stat()
        self.draw_map()

    def run(self, frames, interval=100, animation=False, gif=None):
        if gif != None:
            self.fig, self.ax = plt.subplots()
            anim = FuncAnimation(
                self.fig,
                self.update,
                interval=interval,
                frames=frames,
                repeat=False,
                blit=True,
            )
            anim.save(filename=gif)
            return None
        elif animation:
            self.fig, self.ax = plt.subplots()
            return FuncAnimation(
                self.fig,
                self.update,
                interval=interval,
                frames=frames,
                repeat=False,
                blit=True,
            )
        else:
            for i in range(frames):
                self.update_map()
                self.stat()
            return None

    def print_stat(self):
        print(f"velocity mean: {np.array(self.stat_velocity_mean).mean()} km/h")
        print(f"velocity mean of std: {np.array(self.stat_velocity_std).mean()} km/h")
        print(f"velocity max of std: {np.array(self.stat_velocity_std).max()} km/h")
        print(f"velocity min of std: {np.array(self.stat_velocity_std).min()} km/h")

    def plot_velocity(self):
        # 現在までの統計を出力
        x = np.arange(self.tick)
        sfig, sax = plt.subplots()
        mean = np.array(self.stat_velocity_mean)
        std = np.array(self.stat_velocity_std)
        sax.plot(x, mean, c="green")
        sax.plot(x, mean + std, c="yellow")
        sax.plot(x, mean - std, c="yellow")
        sax.fill_between(x, mean, mean + std, fc="yellow", alpha=0.5)
        sax.fill_between(x, mean, mean - std, fc="yellow", alpha=0.5)

    def plot_seq_mean(self):
        # 現在までの暫定平均値の推移をプロット
        x = np.arange(self.tick)
        fig, ax = plt.subplots()
        ax.plot(x, self.stat_seq_mean)

    def get_latest_mean(self):
        return self.stat_seq_mean[-1]


# 信号を制御できるモデル
class grid_control_model(grid_model):
    def __init__(self, signal_init, map_size=80, seed=0) -> None:
        super().__init__(signal_init.shape[0], map_size, seed)
        self.signal_init = signal_init
        self.make_controler(self.signal_init)

    def make_controler(self, signal_info):
        self.control_signal_list = []
        for i in range(self.n):
            for j in range(self.n):
                signal_list = []
                signal_list.append(self.street_list[Direction.UP][j].signal_list[i])
                signal_list.append(self.street_list[Direction.DOWN][j].signal_list[i])
                signal_list.append(self.street_list[Direction.LEFT][i].signal_list[j])
                signal_list.append(self.street_list[Direction.RIGHT][i].signal_list[j])
                self.control_signal_list.append(
                    ControlSignal(
                        signal_list,
                        signal_info[i][j][0],
                        signal_info[i][j][1],
                        signal_info[i][j][2],
                    )
                )

    # オーバーライド
    def update_signal(self):
        for cntl_signal in self.control_signal_list:
            cntl_signal.update_state()  # ControlSignalオブジェクトによる色制御


# 道(direction, index)と何ブロック目か
class ShoppingMall:
    def __init__(
        self,
        direction: Direction,
        index: int,
        block: int,
        map_size: float,
        n: int,
        color="#999999",
        name="mall",
        name_color="#222222",
    ) -> None:
        self.color = color
        self.size = (map_size / n - ROAD_WIDTH * 2) / 2
        self.direction = direction
        self.map_size = map_size
        self.n = n
        self.name = name
        self.name_color = name_color

        # x,yは中心座標
        if direction == Direction.UP:
            self.x = map_size / n * (index + 1) - ROAD_WIDTH * 2 - self.size / 2
            self.y = map_size / n * (block + 1) - (map_size / n - ROAD_WIDTH * 2) / 2
            self.center_point = (self.x, self.y)
            self.x += self.size / 2 + ROAD_WIDTH
        elif direction == Direction.DOWN:
            self.x = map_size / n * (index + 1) + self.size / 2
            self.y = map_size / n * (block + 1) - (map_size / n - ROAD_WIDTH * 2) / 2
            self.center_point = (self.x, self.y)
            self.x -= self.size / 2 + ROAD_WIDTH
        elif direction == Direction.LEFT:
            self.x = map_size / n * block + (map_size / n - ROAD_WIDTH * 2) / 2
            self.y = map_size / n * (index) - self.size / 2
            self.center_point = (self.x, self.y)
            self.y += self.size / 2 + ROAD_WIDTH
        elif direction == Direction.RIGHT:
            # self.x=map_size/n*block+map_size/n
            self.x = map_size / n * block + (map_size / n - ROAD_WIDTH * 2) / 2
            self.y = map_size / n * (index) + self.size / 2 + ROAD_WIDTH * 2
            self.center_point = (self.x, self.y)
            self.y -= self.size / 2 + ROAD_WIDTH
        else:
            print("ERROR:direction is wrong.")

    def draw(self, ax):
        x_left = self.center_point[0] - self.size / 2
        y_bottom = self.center_point[1] - self.size / 2
        x_left = (x_left + self.map_size) % self.map_size
        y_bottom = (y_bottom + self.map_size) % self.map_size
        ax.add_patch(
            patches.FancyBboxPatch(
                xy=(x_left, y_bottom),
                width=self.size * 0.95,
                height=self.size * 0.95,
                boxstyle="round4",
                mutation_scale=2,
                fc=self.color,
                ec="#000000",
                fill=True,
            )
        )
        fontsize = 8.0
        ax.text(
            x_left,
            y_bottom + self.size / 4,
            self.name,
            fontsize=fontsize,
            color=self.name_color,
        )


# ショッピングモールを1つ持つマップモデル
class grid_mall_model(grid_control_model):
    def __init__(
        self,
        signal_init,
        map_size=80,
        color_mall_first="#999999",
        color_mall_second="#222222",
        seed=0,
    ) -> None:
        super().__init__(signal_init, map_size, seed)
        # 右方向のstreet一番目の1block目にショッピングモールを配置
        city_size_temp = signal_init.shape[0]
        self.shopping_mall = []
        # 1st Mall
        self.shopping_mall.append(
            ShoppingMall(
                Direction.RIGHT,
                1,
                1,
                self.map_size,
                self.n,
                color=color_mall_first,
                name="Mall 1",
                name_color=color_mall_second,
            )
        )
        # 2nd Mall
        self.shopping_mall.append(
            ShoppingMall(
                Direction.DOWN,
                int(city_size_temp / 2),
                int((city_size_temp + 1) / 2),
                self.map_size,
                self.n,
                color=color_mall_second,
                name="Mall 2",
                name_color=color_mall_first,
            )
        )

    # 存在する車のうち一定数ショッピングモールに向かうように設定する。
    def random_change_car_for_SM(self, ratio: float):
        if ratio < 0.0 or 1.0 < ratio:
            print("ERROR: ratio is wrong.")
            return

        all_car_list = []
        for streets in self.street_list:
            for street in streets:
                for car in street.car_list:
                    all_car_list.append(car)

        len_mall_list = len(self.shopping_mall)
        for rand_car in random.sample(all_car_list, int(len(all_car_list) * ratio)):
            mall = self.shopping_mall[randint(0, len_mall_list - 1)]
            rand_car.set_goalpoint(mall.x, mall.y, mall.direction)
            rand_car.color = mall.color

    # 住宅街から出発する車を生成
    def random_make_car_for_path_bedtown(self, n, round_trip=False):
        count = n
        while count:
            res = self.make_car_for_path(
                Direction.RIGHT,  # RIGHT
                0,  # 0番目
                random.random() * self.map_size,
                randint(0, 3),
                randint(0, self.n - 1),
                random.random() * self.map_size,
                round_trip=round_trip,
            )
            if res:
                count -= 1

    # オーバーライド
    def draw_map(self):
        super().draw_map()
        for mall in self.shopping_mall:
            mall.draw(self.ax)


# 信号機制御パラメータ（赤信号の期間、緑信号の期間、位相）の整数インデックスから、実数の信号機制御パラメータに変換するクラス。
class SignalController:
    def __init__(self, city_size, min_sec=1, max_sec=20, nbins_sec=20) -> None:
        self.city_size = city_size

        # 赤信号の表示秒数が取りうる最小値、最大値、離散点数
        self.min_red_duration = min_sec
        self.max_red_duration = max_sec
        self.nbins_red_duration = nbins_sec

        # 青信号の表示秒数が取りうる最小値、最大値、離散点数
        self.min_green_duration = min_sec
        self.max_green_duration = max_sec
        self.nbins_green_duration = nbins_sec

        # 信号の位相が取りうる最小値、最大値、離散点数
        self.min_phase = min_sec
        self.max_phase = max_sec
        self.nbins_phase = nbins_sec

        # 上記の最小値、最大値、離散点数により構築される、信号機制御パラメータの整数インデックステーブル
        self.red_duration = np.linspace(
            self.min_red_duration, self.max_red_duration, self.nbins_red_duration
        )
        self.green_duration = np.linspace(
            self.min_green_duration, self.max_green_duration, self.nbins_green_duration
        )
        self.phase = np.linspace(self.min_phase, self.max_phase, self.nbins_phase)

    def show_index_table(self):
        print("[Index table]")
        nbins_max = max(
            self.nbins_red_duration, self.nbins_green_duration, self.nbins_phase
        )
        df = pd.DataFrame(index=[], columns=range(nbins_max))
        df.loc["Red[i] (sec)"] = np.append(
            np.round(self.red_duration, decimals=2),
            np.full(nbins_max - self.nbins_red_duration, "-"),
        )
        df.loc["Green[i] (sec)"] = np.append(
            np.round(self.green_duration, decimals=2),
            np.full(nbins_max - self.nbins_green_duration, "-"),
        )
        df.loc["Phase[i] (sec)"] = np.append(
            np.round(self.phase, decimals=2),
            np.full(nbins_max - self.nbins_phase, "-"),
        )
        display(df)

    # 信号機の台数分 (city_size*city_size) の信号機制御パラメータ（赤信号の期間、緑信号の期間、位相）の整数インデックスから、実数の信号機制御パラメータに変換する関数
    def decode(self, index_list, disp_flag=False):
        signal_info = []
        for i in range(self.city_size):
            tmp_list = []
            for j in range(self.city_size):
                signal_offset = (i * self.city_size + j) * 3
                # 信号機 [i, j] の赤信号の秒数値、青信号の秒数値、位相の秒数値を追加
                tmp_list.append(
                    (
                        self.red_duration[index_list[signal_offset]],  # 赤信号の秒数
                        self.green_duration[index_list[signal_offset + 1]],  # 青信号の秒数
                        self.phase[index_list[signal_offset + 2]],
                    )
                )  # 位相の秒数
            signal_info.append(tmp_list)
        if disp_flag:
            print("signal_info=[", end="")
            for i in range(len(signal_info) - 1):
                print(f"{signal_info[i]},")
            print(f"{signal_info[len(signal_info) - 1]}]")
        return np.array(signal_info)

    def get_signal_info_random(self, disp_flag=False):
        index_list = np.random.randint(0, 20, self.city_size * self.city_size * 3)
        return self.decode(index_list=index_list, disp_flag=disp_flag)


# ショッピングモール2つの交通シミュレーションクラス
class TrafficSimulator2Malls:
    def __init__(self, signal_info, num_cars, ratio_mall, seed=0):
        self.traffic_sim = grid_mall_model(signal_info, map_size=80, seed=seed)
        self.traffic_sim.random_make_car_for_path(
            num_cars, round_trip=True
        )  # round_tripで往復する車を生成
        self.traffic_sim.random_change_car_for_SM(ratio_mall)

    def run(self, steps, interval=100, animation=False, gif=None):
        return self.traffic_sim.run(
            frames=steps, interval=interval, animation=animation, gif=gif
        )

    def get_latest_mean(self):
        return self.traffic_sim.get_latest_mean()
