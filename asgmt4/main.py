"""Two techniques are presented for consideration, namely post-processing the output of the solver and introducing an error term in the objective value. The approach yielding the highest precision is chosen, that is, post-processing. It is important to note that both methods adhere to the established error bounds."""

import typing
import logging
import math

import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#######
# DATA, do not change this part!
#######
a = [0.5, -0.5, 0.2, -0.7, 0.6, -0.2, 0.7, -0.5, 0.8, -0.4]
l = [40, 20, 40, 40, 20, 40, 30, 40, 30, 60]
Preq = np.arange(a[0], a[0] * (l[0] + 0.5), a[0])
for i in range(1, len(l)):
    Preq = np.r_[Preq, np.arange(Preq[-1] + a[i], Preq[-1] + a[i] * (l[i] + 0.5), a[i])]

T = sum(l)

Peng_max = 20.0
Pmg_min = -6.0
Pmg_max = 6.0
eta = 0.1
gamma = 0.1
#####
# End of DATA part
#####

MAX_OBJ_ERR = 0.1
MAX_CONST_ERR = 0.0003

# Implement the following functions
# they should return a dictionary retval such that
# retval['Peng'] is a list of floats of length T such that retval['Peng'][t] = P_eng(t+1) for each t=0,...,T-1
# retval['Pmg'] is a list of floats of length T such that retval['Pmg'][t] = P_mg(t+1) for each t=0,...,T-1
# retval['Pbr'] is a list of floats of length T such that retval['Pbr'][t] = P_br(t+1) for each t=0,...,T-1
# retval['E'] is a list of floats of length T+1 such that retval['E'][t] = E(t+1) for each t=0,...,T


def _solve(
    e_batt_max: float, epsilon: float = 0.0
) -> typing.Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Variable, cp.Variable]:
    """Solves the minimization problem for the test track fuel consumption.

    Notes
    ----------
    In the initial scenario, the equation E(t+1) = E(t) - Pmg(t) - η|Pmg(t)| applies for t values ranging from 1 to T. However, in DCP, it is not feasible to establish a lower boundary for the distance (as the feasible region is not convex), which results in the relaxation of the constraint to an inequality. It is worth noting that any objective value attained from the original problem can also be reached by the relaxed problem, and vice versa. In order to transform any optimal feasible solution derived from the relaxed problem to a feasible solution of the initial problem, check the "_post_process" function.

    Parameters
    ----------
    e_batt_max : `float`
        Maximum battery capacity. For battery-less vehicles, set to 0.

    epsilon : `float`, optional
        The penalty term in the objective function, by default `0.0` (i.e., w/o penalty)

    Returns
    -------
    `typing.Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Variable, cp.Variable]`
        prob, p_eng, p_mg, p_br, e
    """
    # variables
    p_eng = cp.Variable(T)
    p_mg = cp.Variable(T)
    p_br = cp.Variable(T)
    e = cp.Variable(T + 1)
    p = gamma * np.identity(T)

    # objective
    obj = cp.sum(p_eng + gamma * cp.square(p_eng) + epsilon * cp.maximum(0, -p_mg))

    # constraints
    eng_constraints = [p_eng >= 0, p_eng <= Peng_max]
    mg_constraints = [p_mg >= Pmg_min, p_mg <= Pmg_max]
    br_constraints = [p_br >= 0]
    conserv_constraints = [Preq == p_eng + p_mg - p_br]
    e_constraints = [e[T] == e[0]]
    for j in range(T + 1):
        e_constraints.extend([0 <= e[j], e[j] <= e_batt_max])
        if j == 0:
            continue

        # norm can only be upper bounded
        e_constraints.append(e[j] <= e[j - 1] - p_mg[j - 1] - eta * cp.abs(p_mg[j - 1]))

    constraints = [
        *eng_constraints,
        *mg_constraints,
        *br_constraints,
        *conserv_constraints,
        *e_constraints,
    ]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver="ECOS")
    return prob, p_eng, p_mg, p_br, e


def _post_process(
    p_eng: cp.Variable,
    p_mg: cp.Variable,
    p_br: cp.Variable,
    e: cp.Variable,
    e_batt_max: float,
):
    """Post processes the solver output so the constraints have no slack and are all satisfied tightly.

    Notes
    ----------
    First, the energy conservation constraints are established as an equality, ensuring that all power is transferred to the battery. In case the battery level surpasses the `e_batt_max` threshold, the `p_mg` variable is subjected to mutation such that `e` is set to `e_batt_max`. To maintain the objective value, the car brake is utilized, causing `p_br` to change with the same magnitude of change in `p_mg`.

    Parameters
    ----------
    p_eng : `cp.Variable`

    p_mg : `cp.Variable`

    p_br : `cp.Variable`

    e : `cp.Variable`

    e_batt_max : `float`
    """
    for ix in range(1, len(e.value)):
        act = e.value[ix - 1] - p_mg.value[ix - 1] - eta * abs(p_mg.value[ix - 1])
        if act <= e_batt_max:
            e.value[ix] = act
            continue
        e.value[ix] = e_batt_max

        new_mg = (e.value[ix - 1] - e_batt_max) / (
            1 + math.copysign(eta, p_mg.value[ix - 1])
        )

        mg = p_mg.value[ix - 1]
        delta = new_mg - mg
        p_mg.value[ix - 1] = new_mg
        p_br.value[ix - 1] += delta


def _propagate_retval(
    e: cp.Variable, p_eng: cp.Variable, p_mg: cp.Variable, p_br: cp.Variable
) -> typing.Dict[str, typing.List[float]]:
    retval = {}
    retval["E"] = [float(e.value[ix]) for ix in range(len(e.value))]
    retval["Peng"] = [float(v) for v in p_eng.value]
    retval["Pmg"] = [float(v) for v in p_mg.value]
    retval["Pbr"] = [float(v) for v in p_br.value]
    return retval


def _check_for_glitch(
    e: cp.Variable, p_mg: cp.Variable, new_obj: float, prev_obj: float
):
    """Given the output of a solver, checks if the errors for the objective and relaxed constraints are within the given bounds. If not, raises an exception."""
    const_err = [
        abs(
            (
                e.value[i]
                - e.value[i - 1]
                + p_mg.value[i - 1]
                + eta * abs(p_mg.value[i - 1])
            )
        )
        for i in range(1, T + 1)
    ]
    const_err = max(const_err)
    if const_err > MAX_CONST_ERR:
        raise Exception(
            f"Constraint error is {const_err}. More than {MAX_CONST_ERR} expected."
        )

    obj_err = abs(new_obj - prev_obj)
    if obj_err > MAX_OBJ_ERR:
        raise Exception(
            f"Objective error is {obj_err}. More than {MAX_CONST_ERR} expected."
        )

    logging.info(
        f"Objective and constraint errors are within bound. \n Objective error: {obj_err} \n Constraint error: {const_err}"
    )


def solve_car_cp_pp(e_batt_max: float) -> typing.Dict[str, typing.List[float]]:
    prob, p_eng, p_mg, p_br, e = _solve(e_batt_max)
    prev_obj = prob.objective.value
    _post_process(p_eng, p_mg, p_br, e, e_batt_max)
    _check_for_glitch(e, p_mg, prob.objective.value, prev_obj)
    retval = _propagate_retval(e, p_eng, p_mg, p_br)
    return retval, prob.objective.value


def solve_car_cp_eps(
    e_batt_max: float, eps: float, no_eps_obj: float
) -> typing.Tuple[typing.Dict[str, typing.List[float]], float]:
    """As an alternative to the post-processing approach, a penalty term is incorporated into the objective function to deter the motor/generator from absorbing power if it is not utilized for recharging the battery.

    Parameters
    ----------
    e_batt_max : `float`

    eps : `float`
        Determines the penalty term's weight, small, positive.

    no_eps_obj : `float`

    Returns
    -------
    `typing.Tuple[typing.Dict[str, typing.List[float]], float]`
        retval, prob.objective.value
    """
    prob, p_eng, p_mg, p_br, e = _solve(e_batt_max, eps)
    _check_for_glitch(e, p_mg, prob.objective.value, no_eps_obj)
    retval = _propagate_retval(e, p_eng, p_mg, p_br)
    return retval, prob.objective.value


def car_with_battery():
    Ebatt_max = 100.0
    post_processed, pp_obj = solve_car_cp_pp(Ebatt_max)
    with_eps, eps_obj = solve_car_cp_eps(Ebatt_max, 0.0002, pp_obj)
    return post_processed


def car_without_battery():
    Ebatt_max = 0
    post_processed, pp_obj = solve_car_cp_pp(Ebatt_max)
    with_eps, eps_obj = solve_car_cp_eps(Ebatt_max, 0.0002, pp_obj)
    return post_processed
