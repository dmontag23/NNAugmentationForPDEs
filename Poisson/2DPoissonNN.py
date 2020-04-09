import numpy as np
from datetime import datetime
from fenics import *
from fenics_adjoint import *
from Poisson.FEM import FEM
from Poisson.NN import Network
from scipy.optimize import minimize

np.random.seed(2)

def create_fenics_vector(update_vector):
    q_function = Function(Q)
    vertex_to_dofs = vertex_to_dof_map(q_function.function_space())
    for i in range(len(vertex_to_dofs)):
        q_function.vector()[vertex_to_dofs[i]] = update_vector[i]
    return q_function

def process_nn_parameters(params):
    net.update_parameters(params)
    output = net(mesh.coordinates())
    q_vector = output.detach().numpy()
    q_function = create_fenics_vector(q_vector)
    return output, q_function

def assemble_functional(params):
    output, q = process_nn_parameters(params)
    FEMProblem.q = q
    FEMProblem.solve_poisson_problem()
    J = FEMProblem.error_functional(uex_with_noise, qex)
    return output, q, J

def fun_to_minimize(params):
    output, q_function, J = assemble_functional(params)
    return float(J)

def jac_of_fun(params):
    output, q_function, J = assemble_functional(params)
    control = Control(q_function)
    dJdq = interpolate(compute_gradient(J, control), Q).compute_vertex_values()
    net.calculate_gradients(dJdq, output)
    gradients_at_xi = np.concatenate([param.grad.numpy().flatten() for param in net.parameters()])
    return gradients_at_xi

# ask user which "q" value they would like to observe, what noise level they want, and what value to use for Tikhonov regularization
usr_input = int(input("Please enter a number corresponding to the value of q you want:\n " +
              "0: q = 1 \n 1: q = 1 + x + y \n 2: q = 1 + x^2 + y^2 \n 3: 1 + 0.5 * sin(2*pi*x) * sin(2*pi*y) \n"))
delta = float(input("Please enter a number between 0 and 1 corresponding to the value of noise you want:\n"))
alpha = Constant(float(input("Please enter Tikhonov regularization constant:\n")))

# store the exact value of q and f for the 4 different options for q
all_qex =     [Constant(1.0),
               Expression('1 + x[0] + x[1]', degree=2),
               Expression('1 + pow(x[0], 2) + pow(x[1], 2)', degree=2),
               Expression('1 + 0.5 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])', degree=2)]
all_f   =     [Expression('2 * pow(pi, 2) * sin(pi * x[0]) * sin(pi * x[1])', degree=2),
               Expression('-pi * sin(pi * x[1]) * (cos(pi * x[0]) - pi * (x[0] + x[1] + 1) * sin(pi * x[0])) - pi * sin(pi * x[0]) * (cos(pi * x[1]) - pi * (x[0] + x[1] + 1) * sin(pi * x[1]))', degree=2),
               Expression('-pi * sin(pi * x[1]) * (2 * x[0] * cos(pi * x[0]) - pi * (pow(x[0], 2) + pow(x[1], 2) + 1) * sin(pi * x[0])) - pi * sin(pi * x[0]) * (2 * x[1] * cos(pi * x[1]) - pi * (pow(x[0], 2) + pow(x[1], 2) + 1) * sin(pi * x[1]))', degree=2),
               Expression('-sin(pi * x[0]) * sin(pi * x[1]) * (-2* pow(pi, 2) * sin(2 * pi * x[0]) * sin(2 * pi * x[1]) - (2 * pow(pi, 2))) - (2 * pow(pi, 2)) * pow(cos(pi * x[0]), 3) * pow(sin(pi * x[1]), 2) * cos(pi * x[1]) - (2 * pow(pi, 2)) * pow(sin(pi * x[0]), 2) * cos(pi * x[0]) * pow(cos(pi * x[0]), 3)', degree=2)]

# create the neural network and initialize the weights to be used
net = Network(2, 10)
initial_params = np.random.rand(41)
initial_params[20:30] = 0.0
initial_params[40] = 0.0

# Create mesh and define the function spaces
mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, 'P', 1)
Q = FunctionSpace(mesh, 'P', 1)

net.update_parameters(initial_params)

# define the exact solution for u and q
uex = interpolate(Expression('sin(pi * x[0]) * sin(pi * x[1])', degree=2), V)
qex = interpolate(all_qex[usr_input], Q)

# setup the FEM problem to be solved
f = all_f[usr_input]
FEMProblem = FEM(mesh, V, f, DirichletBC(V, uex, "on_boundary"), delta, alpha)

# find the optimal value for q
uex_with_noise = FEMProblem.add_noise(uex) # add noise to the original solution u to simulate noisy data

net.print_params()
startTime = datetime.now()
min_params = minimize(fun_to_minimize, initial_params, method='BFGS', jac=jac_of_fun, options={'disp':True, 'gtol':1e-07})
print("The time it took to optimize the functional: " + str(datetime.now() - startTime))
output, q_function = process_nn_parameters(min_params.x)
q_opt = interpolate(q_function, Q)

# solve the Poisson problem with the optimal value of q
FEMProblem.q = q_opt
FEMProblem.solve_poisson_problem()

# print the errors
FEMProblem.print_L2_errors(uex, qex)

# write the optimal value of q to a vtk file
vtkfile = File('poisson2d/optq.pvd')
vtkfile << q_opt
vtkfile = File('poisson2d/optu.pvd')
vtkfile << FEMProblem.u