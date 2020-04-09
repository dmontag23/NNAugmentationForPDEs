import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from fenics import *
from fenics_adjoint import *

class FEM:

    def __init__(self, mesh, V, f, bc, delta, alpha, q = None):
        self.__mesh = mesh
        self.__V = V
        self.q = q
        self.__f = f
        self.__bc = bc
        self.u = Function(V)
        self.__delta = delta
        self.__alpha = alpha

    # solve the poisson problem and put the solution in the class variable u
    def solve_poisson_problem(self):
        v = TestFunction(self.__V)
        F = (self.q * inner(grad(self.u), grad(v)) - self.__f * v) * dx
        solve(F == 0, self.u, self.__bc)

    # add noise to the observed data
    def add_noise(self, input_function):
        np.random.seed(2)
        f = interpolate(input_function, input_function.function_space())
        N = f.vector().size()
        f.vector()[:] += self.__delta * np.random.normal(0.0, 1.0, N)  # add noise based off of the normal distribution
        self.__bc.apply(f.vector())  # reset the boundary nodes to be the exact values
        return f

    # define the error functional to be minimized
    def error_functional(self, target_u, target_q):
        J = assemble((0.5 * inner(self.u - target_u, self.u - target_u)) * dx(self.__mesh) +
                     (self.__alpha / 2) *  inner(self.q - target_q, self.q - target_q) * dx(self.__mesh))
        return J

    # minimize the functional
    def minimize_error_functional(self, Jhat):
        startTime = datetime.now()
        q_opt = minimize(Jhat, method = 'BFGS', options={"disp": True, "gtol" : 1e-6})
        print("The time it took to optimize the functional: " + str(datetime.now() - startTime))
        return q_opt

    # print the errors of u and q in the L2 norm
    def print_L2_errors(self, target_u_vector, target_q_vector):
        error_u_L2 = errornorm(self.u, target_u_vector, 'L2')
        print('The error of u in the L2 norm is: ' + str(error_u_L2))
        error_q_L2 = errornorm(self.q, target_q_vector, 'L2')
        print('The error of q in the L2 norm is: ' + str(error_q_L2))

     # plot the solution u(x) and the optimal value of q(x) for the 1D case
    def plot_1D_results(self, uex, qex):
        # plot u(x) using the optimal value of q(x)
        x_values =  self.__mesh.coordinates()[:]
        y_values_FEM = self.u.compute_vertex_values()[:]
        y_actual_values = uex.compute_vertex_values()[:]
        fig, ax = plt.subplots()
        ax.plot(x_values, y_actual_values, marker='.', color='y')
        ax.plot(x_values, y_values_FEM)
        ax.set(xlabel='x', ylabel='u(x)', title='Actual vs FEM Value of u(x), δ=' + str(self.__delta))
        ax.legend(['Actual', 'FEM'], loc='upper right')
        plt.show()

        # plot the optimal value of q(x)
        x_values = self.__mesh.coordinates()[:]
        y_values_FEM = self.q.compute_vertex_values()[:]
        y_actual_values = qex.compute_vertex_values()[:]
        fig, ax = plt.subplots()
        ax.plot(x_values, y_actual_values, marker='.', color='y')
        ax.plot(x_values, y_values_FEM)
        ax.set(xlabel='x', ylabel='q(x)', title='Actual vs FEM Value of q(x), δ=' + str(self.__delta))
        ax.legend(['Actual', 'FEM'], loc='lower right')
        plt.show()
        # fig.savefig('1DPoissonTikGraphs/1D_Pred_Q_FEM_' + str(usr_input) + '_Noise_' + str(self.__delta) + '.jpeg',  bbox_inches='tight')