from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver('GLOP')
w = solver.NumVar(0, solver.infinity(), 'w')
b = solver.NumVar(0, solver.infinity(), 'b')
r = solver.NumVar(0, solver.infinity(), 'r')
c = solver.NumVar(0, solver.infinity(), 'c')
s = solver.NumVar(0, solver.infinity(), 's')


solver.Add(w+b+r+c+s <= 1200)
solver.Add(r <= 250)
solver.Add(s <= 500)
solver.Add(70 * s + 50 * c + 60 * r + 40 * b + 30 * w <= 75000)
solver.Maximize((200-30) * w + (500-40) * b + (800-60)
                * r + (600-50) * c + (900-70)*s)
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('w =', w.solution_value())
    print('b =', b.solution_value())
    print('r =', r.solution_value())
    print('c =', c.solution_value())
    print('s =', s.solution_value())
else:
    print('The problem does not have an optimal solution.')
