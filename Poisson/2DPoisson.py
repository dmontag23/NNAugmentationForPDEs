import matplotlib.pyplot as plt
import numpy as np
from fenics import *
from fenics_adjoint import *

# ask user which "q" value they would like to observe
usr_input = input("Please enter a number corresponding to the value of q you want:\n " +
              "0: q = 1 \n 1: q = 1 + x + y \n 2: q = 1 + x^2 + y^2 \n 3: 1 + 0.5 * sin(2*pi*x) * sin(2*pi*y) \n")
usr_input = int(usr_input)

# store the exact value of q and f and the initial guess of q for the 4 different options for q
all_qex =    [Constant(1.0),
              Expression('1 + x[0] + x[1]', degree=2),
              Expression('1 + pow(x[0], 2) + pow(x[1], 2)', degree=2),
              Expression('1 + 0.5 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])', degree=2)]
all_f   =    [Expression('2 * pow(pi, 2) * sin(pi * x[0]) * sin(pi * x[1])', degree=2),
              Expression('-pi * sin(pi * x[1]) * (cos(pi * x[0]) - pi * (x[0] + x[1] + 1) * sin(pi * x[0])) - pi * sin(pi * x[0]) * (cos(pi * x[1]) - pi * (x[0] + x[1] + 1) * sin(pi * x[1]))', degree=2),
              Expression('-pi * sin(pi * x[1]) * (2 * x[0] * cos(pi * x[0]) - pi * (pow(x[0], 2) + pow(x[1], 2) + 1) * sin(pi * x[0])) - pi * sin(pi * x[0]) * (2 * x[1] * cos(pi * x[1]) - pi * (pow(x[0], 2) + pow(x[1], 2) + 1) * sin(pi * x[1]))', degree=2),
              Expression('-sin(pi * x[0]) * sin(pi * x[1]) * (-2* pow(pi, 2) * sin(2 * pi * x[0]) * sin(2 * pi * x[1]) - (2 * pow(pi, 2))) - (2 * pow(pi, 2)) * pow(cos(pi * x[0]), 3) * pow(sin(pi * x[1]), 2) * cos(pi * x[1]) - (2 * pow(pi, 2)) * pow(sin(pi * x[0]), 2) * cos(pi * x[0]) * pow(cos(pi * x[0]), 3)', degree=2)]
all_qguess = [Expression('0.25', degree=2),
              Expression('0.25 + x[0] + x[1]', degree=2),
              Expression('0.25 + 0.25 * pow(x[0], 2) + 0.25 * pow(x[1], 2)', degree=2),
              Expression('0.25 + 0.25 * sin(x[0]) * sin(x[1])', degree=2)]

# set the exact value of u and q
uex = Expression('sin(pi * x[0]) * sin(pi * x[1])', degree=2)
qex = all_qex[usr_input]

# Create mesh and define the function spaces
mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, 'P', 1)
Q = FunctionSpace(mesh, 'DG', 0)

# Define boundary condition
u_D = uex
bc = DirichletBC(V, u_D, "on_boundary")

# add noise to the observed data
np.random.seed(2)
uex = interpolate(uex, V)
N = uex.vector().size()
delta = 0.0
uex.vector()[:] += delta * np.random.normal(0.0, 1.0, N)  # add noise based off of the normal distribution
# reset the boundary nodes to be the exact values
bc.apply(uex.vector())

# plot the observed data with noise
plot(uex, title="Observed Data with Noise")
plt.show()

# initial guess for q
q = interpolate(all_qguess[usr_input], Q)

# Define variational problem
u = Function(V)
v = TestFunction(V)
f = all_f[usr_input]
F = (q * inner(grad(u), grad(v)) - f * v) * dx
solve(F == 0, u, bc)

# Define the error functional to be minimized
alpha = Constant(0.0) # Do not use any Tikhonov regularization - change this to a positive number to use regularization of the error functional
J = assemble((0.5 * inner(u - uex, u - uex)) * dx(mesh) + alpha / 2 *  inner(q - qex, q - qex) * dx(mesh))
control = Control(q)
Jhat = ReducedFunctional(J, control)
ReducedFunctionalTorch

# minimize the functional
g_opt = minimize(Jhat, method = 'L-BFGS-B', options={"disp": True, "gtol" : 1e-7})

# write the optimal value of q to a vtk file
vtkfile = File('poisson2d/optq.pvd')
vtkfile << g_opt