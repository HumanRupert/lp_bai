# Import packages. You can import additional packages, if you want
# You can change the way they are imported, e.g import pulp as pl or whatever
# But take care to adapt the solver configuration accordingly.
import typing as T
import itertools
import time

from pulp import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

# Use the following solver
solver = COIN_CMD(path="/usr/bin/cbc", threads=8)

###################
## !COMMENT OUT! ##
##################
# solver = getSolver("PULP_CBC_CMD")
# at home, you can try it with different solvers, e.g. GLPK, or with a different
# number of threads.
# WARNING: your code when run in vocareum should finish within 10 minutes!!!

M = 10 ** 10


def do_optimize(cakes: T.List[T.List[int]]) -> T.Dict[str, float]:
    # define problem
    prob = LpProblem("The Pastry Problem", LpMinimize)

    # define vars
    starts = [LpVariable(f"s_{datum[0]}", lowBound=0) for datum in cakes]

    # add basic constraints
    for cake in cakes:
        prob += starts[cake[0]] >= cake[1], f"Start after pre, {cake[0]}"
        prob += starts[cake[0]] + \
            cake[3] <= cake[2], f"End before dln, {cake[0]}"

    # add anti parallelism constraints
    for cake1, cake2 in itertools.product(cakes, repeat=2):
        w = LpVariable(f"w_{cake1[0]},{cake2[0]}",
                       lowBound=0, upBound=1, cat=LpInteger)
        z = LpVariable(f"z_{cake1[0]},{cake2[0]}",
                       lowBound=0, upBound=1, cat=LpInteger)
        prob += w+z <= 1
        prob += starts[cake1[0]] + cake1[3] <= starts[cake2[0]] + M * w
        prob += starts[cake2[0]] + cake2[3] <= starts[cake1[0]] + M * z

    prob += lpSum(starts)

    prob.solve()

    return {var_.name: var_.value()
            for var_ in prob.variables()[:len(cakes)]}


def _vis_gantt(cakes: T.List[T.List[int]], ax: plt.Axes):
    cakes = sorted(cakes, key=lambda x: x[-1])
    y = [f"#{d[0]}" for d in cakes]
    width = [d[3] / (60*60) for d in cakes]
    left = [d[4] / (60*60) for d in cakes]
    dln = [d[2] / (60*60) for d in cakes]
    pre = [d[1] / (60*60) for d in cakes]
    ax.barh(y=y, width=width, left=left)
    ax.barh(y=y, width=[0.01] * len(y), left=dln, color="r")
    ax.barh(y=y, width=[0.01] * len(y), left=pre, color="g")
    ax.set_xlabel("Time (Hours)")
    ax.set_ylabel("Cake")
    ax.set_title("Bakery Schedule Gantt Chart")
    leg1 = mlines.Line2D([], [], color='g', marker='|',
                         ls='', label='Dough preparation')
    leg2 = mlines.Line2D([], [], color='r', marker='|',
                         ls='', label='Customer arrival')
    ax.legend(handles=[leg1, leg2])
    ax.plot()


def _vis_pre_buffer(cakes: T.List[T.List[int]], ax: plt.Axes):
    cakes = sorted(cakes, key=lambda x: x[4] - x[1])
    x = [f"#{d[0]}" for d in cakes]
    y = [(cake[4] - cake[1]) / (60 * 60) for cake in cakes]
    ax.bar(x, y)
    ax.set_xlabel("Cake")
    ax.set_ylabel("Time (Hours)")
    ax.set_title("Dough Preparation to Baking Start Buffer")
    ax.text(3.5, 4.5, '(Time from when pastry is ready for baking to starting time of baking)', fontsize=9)
    ax.tick_params(axis='x', rotation=20)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    tights = [x[i] for i in range(len(y)) if y[i] == 0]
    ax.text(
        x=1, y=2, s=f"* Cakes {','.join(tights)} need to be immediately baked once the dough is prepared", bbox=props)
    ax.plot()


def _vis_dln_buffer(cakes: T.List[T.List[int]], ax: plt.Axes):
    cakes = sorted(cakes, key=lambda x: x[2] - (x[4] + x[3]))
    x = [f"#{d[0]}" for d in cakes]
    y = [(cake[2] - (cake[4] + cake[3])) / (60 * 60) for cake in cakes]
    ax.bar(x, y)
    ax.set_xlabel("Cake")
    ax.set_ylabel("Time (Hours)")
    ax.set_title("Baking End to Customer Arrival Buffer")
    ax.text(
        4, 6.3, '(Time from when pastry is ready to be served to the deadline)', fontsize=9)
    ax.tick_params(axis='x', rotation=20)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    tights = [x[i] for i in range(len(y)) if y[i] == 0]
    ax.text(
        x=.5, y=2, s=f"* Cakes {','.join(tights)} must be delivered to the customer immediately after baking \n (no leeway!)", bbox=props)
    ax.plot()


def do_visualize(schedule: T.Dict[str, float], cakes: T.List[T.List[int]], out: str):
    figure, axis = plt.subplots(
        3, 1, gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=(10, 15))
    cakes = [[*cake, schedule[f"s_{cake[0]}"]] for cake in cakes]
    _vis_gantt(cakes, axis[0])
    _vis_pre_buffer(cakes, axis[1])
    _vis_dln_buffer(cakes, axis[2])
    figure.tight_layout()

    plt.savefig(out)


def bakery():
    st = time.time()

    # Input file is called ./bakery.txt
    input_filename = './bakery.txt'
    cakes = [[int(val) for val in line.rstrip().split(" ")]
             for line in open(input_filename, 'r')]

    # Use solver defined above as a parameter of .solve() method.
    # e.g., if your LpProblem is called prob, then run
    # prob.solve(solver) to solve it.
    retval = do_optimize(cakes)

    # Write visualization to the correct file:
    visualization_filename = './visualization.png'
    do_visualize(retval, cakes, visualization_filename)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    # retval should be a dictionary such that retval['s_i'] is the starting
    # time of pastry i
    return retval
