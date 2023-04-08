import typing as T
import networkx as nx
import numpy as np
import os
os.chdir("asgmt2")

"""

The architecture of the solution is inspired by Gyro-Aided Visual Tracking Using Iterative Earth Mover's Distance by Yao et al., Universiy of Conneticut (DOI. No. 10.1109/MAES.2017.160223)
"""

# Here you may include additional libraries and define more auxiliary functions:

X_SIZE = 80


def _read_file(directory: str) -> np.array:
    with open(directory, 'r') as f:
        arr = [[int(num) for num in line if num != "\n"]
               for line in f.readlines()][:10]
        return np.array(arr)


def _lcm_scale(m1: np.array, m2: np.array) -> T.Tuple[np.array, np.array, float]:
    """The sum of the brightness values might not be the same, which means the two distributions might not have the same integral.  We could either transform the distribution with the smaller integral so its sum equates the other. Alternatively, we could scale both into their LCM. While the latter solution might be more computationally intensive, the output will always be an integer. Thus, we use LCMs.

    ###
    The solution was inspired by: Erickson, William. (2020). A generalization for the expected value of the one-dimensional earth mover's distance.
    ###"""
    m1_sum, m2_sum = np.sum(m1), np.sum(m2)
    lcm = np.lcm(m1_sum, m2_sum)
    m1 = (m1 * lcm / m1_sum).astype(int)
    m2 = (m2 * lcm / m2_sum).astype(int)
    return m1, m2, lcm


def _make_graph(m1: np.array, m2: np.array) -> nx.DiGraph:
    """Bright pixels are nodes in the graph (piles), brightness value is the probability mass in each node (weight of the pile), and the number of right-sided steps between two pixels is the cost of the edge (the distance b/w piles), one can think of this as the Manhattan distance where the object can only move in one direction."""
    G = nx.DiGraph()
    it1 = np.nditer(m1, flags=["multi_index"])

    for node1 in it1:
        if not node1:
            continue
        it2 = np.nditer(m2, flags=["multi_index"])
        for node2 in it2:
            if not node2:
                continue
            w1 = int(node1)
            w2 = int(node2)
            n1 = f"m1_{it1.multi_index[0]}_{it1.multi_index[1]}"
            n2 = f"m2_{it2.multi_index[0]}_{it2.multi_index[1]}"
            x1 = it1.multi_index[1]
            x2 = it2.multi_index[1]
            dist = x2 - x1 if x2 >= x1 else X_SIZE + (x2 - x1)
            (n1 not in G.nodes) and G.add_node(n1, demand=w1 * -1)
            (n2 not in G.nodes) and G.add_node(n2, demand=w2)
            G.add_edge(n1, n2, weight=dist, capacity=w1)

    return G

# This function should return the EMD distances between file1 and file2.
# EMD distance depends on your choice of distance between pixels and
# this will be taken into account during grading.


def comp_dist(file1: str, file2: str) -> float:
    m1, m2 = _read_file(file1), _read_file(file2)

    m1, m2, lcm = _lcm_scale(m1, m2)

    G = _make_graph(m1, m2)

    distance = nx.algorithms.min_cost_flow_cost(G) / lcm

    # And return the EMD distance, it should be float.
    return float(distance)

# This function should sort the files as described on blackboard.
# P1.txt should be the first one.


def sort_files():
    # If your code cannot handle all files, remove the problematic ones
    # from the following list and sort only those which you can handle.
    # Your code should never fail!
    files = ['P1.txt', 'P2.txt', 'P3.txt', 'P4.txt', 'P5.txt', 'P6.txt', 'P7.txt',
             'P8.txt', 'P9.txt', 'P10.txt', 'P11.txt', 'P12.txt', 'P13.txt', 'P14.txt', 'P15.txt']
    # Write your code here:

    files_dist = [{"file": file, "dist": comp_dist(
        "P1.txt", file)} for file in files]

    sorted_files = [d["file"] for d in sorted(
        files_dist, key=lambda x: x["dist"])]
    # should return sorted list of file names
    return sorted_files
