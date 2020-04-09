from fenics import *
from fenics_adjoint import *
from Poisson.FEM import FEM

# ask user which "q" value they would like to observe, what noise level they want, and what value to use for Tikhonov regularization
usr_input = int(input("Please enter a number corresponding to the value of q you want:\n " +
              "0: q = 1 \n 1: q = 1 + x + y \n 2: q = 1 + x^2 + y^2 \n 3: 1 + 0.5 * sin(2*pi*x) * sin(2*pi*y) \n"))
delta = float(input("Please enter a number between 0 and 1 corresponding to the value of noise you want:\n"))
alpha = Constant(float(input("Please enter Tikhonov regularization constant:\n")))

# store the exact value of q and f and the initial guess of q for the 4 different options for q
all_qex =     [Constant(1.0),
               Expression('1 + x[0] + x[1]', degree=2),
               Expression('1 + pow(x[0], 2) + pow(x[1], 2)', degree=2),
               Expression('1 + 0.5 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])', degree=2)]
all_f   =     [Expression('2 * pow(pi, 2) * sin(pi * x[0]) * sin(pi * x[1])', degree=2),
               Expression('-pi * sin(pi * x[1]) * (cos(pi * x[0]) - pi * (x[0] + x[1] + 1) * sin(pi * x[0])) - pi * sin(pi * x[0]) * (cos(pi * x[1]) - pi * (x[0] + x[1] + 1) * sin(pi * x[1]))', degree=2),
               Expression('-pi * sin(pi * x[1]) * (2 * x[0] * cos(pi * x[0]) - pi * (pow(x[0], 2) + pow(x[1], 2) + 1) * sin(pi * x[0])) - pi * sin(pi * x[0]) * (2 * x[1] * cos(pi * x[1]) - pi * (pow(x[0], 2) + pow(x[1], 2) + 1) * sin(pi * x[1]))', degree=2),
               Expression('-sin(pi * x[0]) * sin(pi * x[1]) * (-2* pow(pi, 2) * sin(2 * pi * x[0]) * sin(2 * pi * x[1]) - (2 * pow(pi, 2))) - (2 * pow(pi, 2)) * pow(cos(pi * x[0]), 3) * pow(sin(pi * x[1]), 2) * cos(pi * x[1]) - (2 * pow(pi, 2)) * pow(sin(pi * x[0]), 2) * cos(pi * x[0]) * pow(cos(pi * x[0]), 3)', degree=2)]
all_q_guess = [Expression('0.25', degree=2),
               Expression('0.25 + x[0] + x[1]', degree=2),
               Expression('0.25 + 0.25 * pow(x[0], 2) + 0.25 * pow(x[1], 2)', degree=2),
               Expression('0.25 + 0.25 * sin(x[0]) * sin(x[1])', degree=2)]

# Create mesh and define the function spaces
mesh = UnitSquareMesh(100, 100)
Q = FunctionSpace(mesh, 'DG', 0)
V = FunctionSpace(mesh, 'P', 1)

# define the exact solution for u and q, along with the inital guess for q
uex = interpolate(Expression('sin(pi * x[0]) * sin(pi * x[1])', degree=2), V)
qex = interpolate(all_qex[usr_input], Q)
q_guess = interpolate(all_q_guess[usr_input], Q)

# setup the FEM problem to be solved
f = all_f[usr_input]
FEMProblem = FEM(mesh, V, f, DirichletBC(V, uex, "on_boundary"), delta, alpha, q_guess)

# find the optimal value for q
uex_with_noise = FEMProblem.add_noise(uex) # add noise to the original solution u to simulate noisy data
FEMProblem.solve_poisson_problem()
J = FEMProblem.error_functional(uex_with_noise, qex)
q_opt = FEMProblem.minimize_error_functional(ReducedFunctional(J, Control(FEMProblem.q)))

# solve the Poisson problem with the optimal value of q
FEMProblem.q = q_opt
FEMProblem.solve_poisson_problem()

# print the errors
FEMProblem.print_L2_errors(uex, qex)

# write the optimal value of q to a vtk file
vtkfile = File('poisson2d/optq.pvd')
vtkfile << q_opt