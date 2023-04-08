from pulp import *


def ex1():
    retval = {}
    retval["x"] = None
    retval["y"] = None
    retval["obj"] = None
    retval["tight_constraints"] = [None]
    # Insert your code below:
    x = LpVariable("x")
    y = LpVariable("y")
    prob = LpProblem("myProblem", LpMinimize)
    prob += x >= -10
    prob += y <= 10
    prob += 122*x + 143*y
    prob += 3*x + 2*y <= 10
    prob += 12*x + 14*y >= -12.5
    prob += 2*x + 3*y >= 3
    prob += 5*x - 6*y >= -100
    status = prob.solve()
    retval["x"] = value(x)
    retval["y"] = value(y)
    retval["obj"] = prob.objective.value()

    ########
    ########
    ########
    ########
    ########
    # dear professor, if you're reading this, i hope you're having a very nice day
    # if you have a window nearby, please look outside and enjoy the sunlight
    # if there's no sunlight or no windows, then don't. instead, look at these funny animals:
    # https://www.youtube.com/watch?v=WnrjOxxS9wo #
    ########
    ########
    ########
    ########
    ########
    retval["tight_constraints"] = [ix+1 for ix,
                                   (key, val) in enumerate(prob.constraints.items()) if not val.slack]

    return retval


def ex2():
    retval = {}
    retval['x1'] = None
    retval['x2'] = None
    retval['x3'] = None
    retval['x4'] = None
    retval['x5'] = None
    retval['x6'] = None
    retval['obj'] = None

    # Insert your code below:
    x1 = LpVariable("x1", 0, 1)
    x2 = LpVariable("x2", 0, 1)
    x3 = LpVariable("x3", 0, 1)
    x4 = LpVariable("x4", 0, 1)
    x5 = LpVariable("x5", 0, 1)
    x6 = LpVariable("x6", 0, 1)
    x0 = LpVariable("x0", 0)

    game = LpProblem("game", LpMinimize)
    game += x1 + x2 + x3 + x4 + x5 + x6 == 1
    game += 2*x2 - x3 - x4 - x5 - x6 <= x0
    game += -2 * x1 + 2*x3 - x4 - x5 - x6 <= x0
    game += x1 - 2*x2 + 2*x4 - x5 - x6 <= x0
    game += x1 + x2 - 2*x3 + 2*x5 - x6 <= x0
    game += x1 + x2 + x3 - 2*x4 + 2*x6 <= x0
    game += x1 + x2 + x3 + x4 - 2*x5 <= x0
    game += x0
    game.solve()

    retval["obj"], retval["x1"], retval["x2"], retval["x3"], retval["x4"], retval["x5"], retval["x6"] = value(
        x0), value(x1), value(x2), value(x3), value(x4), value(x5), value(x6)
    return retval


def ex3():
    retval = {}
    retval['obj'] = None
    retval['x1'] = None
    # there should be retval['xi'] for each company number i
    # Insert your code below:
    data = open("asgmt1/hw1-03.txt", 'r')

    companies = range(1, 70)
    contracts = [[int(val) for val in line.rstrip().split(" ")]
                 for line in data]

    cover = LpProblem("cover", LpMinimize)
    variables = [LpVariable(f"x{i}", 0, cat="Integer") for i in companies]
    for c in contracts:
        cover += variables[c[0]-1] + variables[c[1]-1] >= 2
    cover += sum(variables)

    cover.solve()

    retval["obj"] = cover.objective.value()
    for var in variables:
        retval[str(var)] = value(var)
    # return retval dictionary
    return retval


ex3()
