from fenics import *
from fenics_adjoint import *
from Poisson.FEM import FEM

# ask user which "q" value they would like to observe, what noise level they want, and what value to use for Tikhonov regularization
usr_input = int(input("Please enter a number corresponding to the value of q you want:\n " +
              "0: q = 1 \n 1: q = 1 + x \n 2: q = 1 + x^2 \n 3: 1 + 0.5 * sin(2*pi*x) \n"))
delta = float(input("Please enter a number between 0 and 1 corresponding to the value of noise you want:\n"))
alpha = Constant(float(input("Please enter Tikhonov regularization constant:\n")))

# store the exact value of q and f and the initial guess of q for the 4 different options for q
all_qex =     [Constant(1.0),
               Expression('1 + x[0]', degree=2),
               Expression('1 + pow(x[0], 2)', degree=2),
               Expression('1 + 0.5 * sin(2 * pi * x[0])', degree=2)]
all_f   =     [Expression('-8 * pow(pi, 2) * cos(4 * pi * x[0])', degree=2),
               Expression('-2 * pi * (sin(4 * pi * x[0]) + 4 * pi * (1 + x[0]) * cos(4 * pi * x[0]))', degree=2),
               Expression('-8 * pow(pi, 2) * (1 + pow(x[0], 2)) * cos(4 * pi * x[0]) - 4 * pi * x[0] * sin(4 * pi * x[0])', degree=2),
               Expression('-2 * pow(pi, 2) * (2 * (sin(2 * pi * x[0]) + 2) * cos(4 * pi * x[0]) + sin(4 * pi * x[0]) * cos(2 * pi * x[0]))', degree=2)]
all_q_guess = [Expression('0.25', degree=2),
               Expression('0.25 + 0.25 * x[0]', degree=2),
               Expression('0.25 + 0.25 * pow(x[0], 2)', degree=2),
               Expression('0.25 + 0.25 * sin(x[0])', degree=2)]

# Create mesh and define the function spaces
mesh = IntervalMesh(100,0,1)
Q = FunctionSpace(mesh, 'DG', 0)
V = FunctionSpace(mesh, 'P', 1)

# define the exact solution for u and q, along with the inital guess for q
uex = interpolate(Expression('pow(sin(2 * pi * x[0]), 2)', degree=2), V)
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

# print the errors and plot the results
FEMProblem.print_L2_errors(uex, qex)
FEMProblem.plot_1D_results(uex_with_noise, qex)