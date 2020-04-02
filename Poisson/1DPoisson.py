import matplotlib.pyplot as plt
import numpy as np
from fenics import *
from fenics_adjoint import *
from datetime import datetime

# ask user which "q" value they would like to observe
usr_input = input("Please enter a number corresponding to the value of q you want:\n " +
              "0: q = 1 \n 1: q = 1 + x \n 2: q = 1 + x^2 \n 3: 1 + 0.5 * sin(2*pi*x) \n")
usr_input = int(usr_input)

# store the exact value of q and f and the initial guess of q for the 4 different options for q
all_qex =    [Constant(1.0),
              Expression('1 + x[0]', degree=2),
              Expression('1 + pow(x[0], 2)', degree=2),
              Expression('1 + 0.5 * sin(2 * pi * x[0])', degree=2)]
all_f   =    [Expression('-8 * pow(pi, 2) * cos(4 * pi * x[0])', degree=2),
              Expression('-2 * pi * (sin(4 * pi * x[0]) + 4 * pi * (1 + x[0]) * cos(4 * pi * x[0]))', degree=2),
              Expression('-8 * pow(pi, 2) * (1 + pow(x[0], 2)) * cos(4 * pi * x[0]) - 4 * pi * x[0] * sin(4 * pi * x[0])', degree=2),
              Expression('-2 * pow(pi, 2) * (2 * (sin(2 * pi * x[0]) + 2) * cos(4 * pi * x[0]) + sin(4 * pi * x[0]) * cos(2 * pi * x[0]))', degree=2)]
all_qguess = [Expression('0.25', degree=2),
              Expression('0.25 + 0.25 * x[0]', degree=2),
              Expression('0.25 + 0.25 * pow(x[0], 2)', degree=2),
              Expression('0.25 + 0.25 * sin(x[0])', degree=2)]

# set the exact value of u and q
uex = Expression('pow(sin(2 * pi * x[0]), 2)', degree=2)
qex = all_qex[usr_input]

# Create mesh and define the function spaces
mesh = IntervalMesh(100,0,1)

V = FunctionSpace(mesh, 'P', 1)
Q = FunctionSpace(mesh, 'DG', 0)

# Define boundary condition
u_D = uex
bc = DirichletBC(V, u_D, "on_boundary")

# add noise to the observed data
np.random.seed(2)
uex_with_noise = interpolate(uex, V)
N = uex_with_noise.vector().size()
delta = 0.00
uex_with_noise.vector()[:] += delta * np.random.normal(0.0, 1.0, N)  # add noise based off of the normal distribution
# reset the boundary nodes to be the exact values
bc.apply(uex_with_noise.vector())

# interpolate the initial guess for q and the exact solution for u and q into the FE space
q = interpolate(all_qguess[usr_input], Q)
uex = interpolate(uex, V)
qex = interpolate(qex, Q)

# Define variational problem
u = Function(V)
v = TestFunction(V)
f = all_f[usr_input]
F = (q * inner(grad(u), grad(v)) - f * v) * dx
solve(F == 0, u, bc)

# Define the error functional to be minimized
alpha = Constant(0.0) # Do not use any Tikhonov regularization - change this to a positive number to use regularization of the error functional
J = assemble((0.5 * inner(u - uex_with_noise, u - uex_with_noise)) * dx(mesh) + (alpha / 2) *  inner(q - qex, q - qex) * dx(mesh))
control = Control(q)
Jhat = ReducedFunctional(J, control)
test = interpolate (Jhat.derivative(), Q)
print(test.vector()[:])

# minimize the functional and print the time it took to do the optimization
startTime = datetime.now()
q_opt = minimize(Jhat, method = 'BFGS', options={"disp": True, "gtol" : 1e-6})
print("The time it took to optimize the functional: " + str(datetime.now() - startTime))

# solve the problem with the optimal value of q found
F = (q_opt * inner(grad(u), grad(v)) - f * v) * dx
solve(F == 0, u, bc)

# plot the solution with optimal value of q(x)
x_values =  mesh.coordinates()[:]
y_values_FEM = u.compute_vertex_values()[:]
y_actual_values = uex_with_noise.compute_vertex_values()[:]
fig, ax = plt.subplots()
ax.plot(x_values, y_actual_values, marker='.', color='y')
ax.plot(x_values, y_values_FEM)
ax.set(xlabel='x', ylabel='u(x)', title='Actual vs FEM Value of u(x), δ=' + str(delta))
ax.legend(['Actual', 'FEM'], loc='upper right')
plt.show()
fig.savefig('1DPoissonTikGraphs/1D_Pred_U_FEM_' + str(usr_input) + '_Noise_' + str(delta) + '.jpeg',  bbox_inches='tight')

# print the error of u and q in the L2 norm
error_uL2 = errornorm(u, uex, 'L2')
error_qL2 = errornorm(q_opt, qex, 'L2')
print('The error of u in the L2 norm is: ' + str(error_uL2))
print('The error of q in the L2 norm is: ' + str(error_qL2))

# plot the optimal value of q(x) given by FEM
x_values = mesh.coordinates()[:]
y_values_FEM = q_opt.compute_vertex_values()[:]
y_actual_values = qex.compute_vertex_values()[:]
fig, ax = plt.subplots()
ax.plot(x_values, y_actual_values, marker='.', color='y')
ax.plot(x_values, y_values_FEM)
ax.set(xlabel='x', ylabel='q(x)', title='Actual vs FEM Value of q(x), δ=' + str(delta))
ax.legend(['Actual', 'FEM'], loc='lower right')
plt.show()
fig.savefig('1DPoissonTikGraphs/1D_Pred_Q_FEM_' + str(usr_input) + '_Noise_' + str(delta) + '.jpeg',  bbox_inches='tight')