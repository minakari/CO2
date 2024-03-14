
# coding: utf-8

# In[301]:


from __future__ import print_function
from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random


# In[303]:


mesh = RectangleMesh(Point(0, 0), Point(200, 100), 60, 30)
mesh_points=mesh.coordinates()

mesh_points_x = mesh.coordinates()[:,0].T
mesh_points_y = mesh.coordinates()[:,1].T
nn = np.shape(mesh_points_x)[0]

points = np.zeros((nn,2))
for i in range (nn):
    points[i,:] = (mesh_points_x[i], mesh_points_y[i])


# In[304]:


E = 250000000
nu = 0.3
lmbda = Constant(E*nu/((1+nu)*(1-2*nu)))
mu = Constant(E/2/(1+nu))


# In[305]:


d = 1 # interpolation degree
Vue = VectorElement('CG', mesh.ufl_cell(), d) # displacement finite element
Vp1e = FiniteElement('CG', mesh.ufl_cell(), d) # concentration finite element
Vp2e = FiniteElement('CG', mesh.ufl_cell(), d) # concentration finite element
Vp3e = FiniteElement('DG', mesh.ufl_cell(), 0) # pressure finite element
Vp4e = FiniteElement('CG', mesh.ufl_cell(), d) # volume fraction finite element
Vp5e = FiniteElement('DG', mesh.ufl_cell(), d) # \psi finite element
V = FunctionSpace(mesh, MixedElement([Vue, Vp1e, Vp2e, Vp3e, Vp4e, Vp5e]))

# Boundary conditions
def bottom(x, on_boundary):
    return near(x[1], 0.0) and on_boundary
def left(x, on_boundary):
    return near(x[0], 0.0) and on_boundary
def right1(x, on_boundary):
    return near(x[0], 200.0) and (x[1]>=20) and on_boundary
def right2(x, on_boundary):
    return near(x[0], 200.0) and (x[1]<20) and on_boundary
def top(x, on_boundary):
    return near(x[1], 100.0) and on_boundary

# co2
bc1 = DirichletBC(V.sub(1), Constant(0.), left)

# brine
bc5 = DirichletBC(V.sub(2), Constant(218.9), left)
bc10 = DirichletBC(V.sub(0).sub(0), Constant(0.), left)
bc12 = DirichletBC(V.sub(0).sub(1), Constant(0.), bottom)

bcs = [bc1,bc5,bc10,bc12]


# In[306]:


# Defining multiple Neumann boundary conditions
mf = MeshFunction("size_t", mesh, 1)
mf.set_all(0) # initialize the function to zero

class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary
bottom = bottom() # instantiate it
bottom.mark(mf, 1)

class top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 100.0) and on_boundary
top = top() # instantiate it
top.mark(mf, 2)

class right1(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 200.0) and (x[1]>=20) and on_boundary
right1 = right1() # instantiate it
right1.mark(mf, 3)

class right2(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 200.0) and (x[1]<20) and on_boundary
right2 = right2() # instantiate it
right2.mark(mf, 4)

class left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary
left = left() # instantiate it
left.mark(mf, 5)

ds = ds(subdomain_data = mf)


# In[312]:


U_ = TestFunction(V)
(u_, P1_, P2_, p_, phic_,psi_) = split(U_)
dU = TrialFunction(V)
(du, dP1, dP2, dp, dphic, dpsi) = split(dU)
U = Function(V)
(u, P1, P2, p, phic, psi) = split(U)  # P1 = \rho_aR and P2 = \rho_wR

Un = Function(V)
(un, Pn1, Pn2, pn, phicn, psin) = split(Un)  # P1 = \rho_aR and P2 = \rho_wR


# In[314]:


d = u.geometric_dimension()
I = Identity(d)             # Identity tensor

F = grad(u)             # Deformation gradient

kappa = 2.0*10**(-12)            # Permeability of soil
vis_w = 0.001
vis_a = 3*10**(-5)

phi = 0.8  # \phi_s
gamma_w = 1100

R = 8.32
T = 280
a = 0.364
b = 0.00004267

# Invariants of deformation tensors
J1 = det(I + grad(u))

phi_w = (P2)/(J1*gamma_w)
epsilon = 0.001
gamma_a = np.abs(P1)/(np.abs(phic)+epsilon)

Ic = tr((F + F.T)/2)
F1 = phi*(lmbda*Ic * I + mu*(F + F.T)) - (1-phi)*p*I

g = Expression(("0.0", "-10.0"), degree=0)

J_a = -(phic/J1)*(kappa/vis_a)*gamma_a*(R*T*grad(gamma_a)/(1-b*gamma_a) + b*R*T*gamma_a*grad(gamma_a)/(1-b*gamma_a)**2 - 2*a*gamma_a*grad(gamma_a) - 0.044*gamma_a*g -0.01*gamma_a*grad(psi)) # P_p1
J_w = -phi_w*(kappa/vis_w)*gamma_w*(grad(p) - gamma_w*g) # P_p2

#  Define time things.
Tt, num_steps = 3600 , 1800
dt = Tt / (1.0*num_steps)

P_init2 = Expression ( "218.9", degree = 0 )
Pn2 = project ( P_init2, V.sub(1).collapse())

p_init = Expression ( "-6000000.0", degree = 0 )
pn = project ( p_init, V.sub(3).collapse())

y_BC = Expression(("0.0", "0.0"), degree=0)
u = project(y_BC,V.sub(0).collapse())

g_a = Expression(("-0.2"), degree=0)
g_w = Expression(("0.0"), degree=0)

mech_form = -inner(F1, grad(u_))*dx + (phi_w*gamma_w)*inner(g,u_)*dx + 0.044*(phic*gamma_a/J1)*inner(g,u_)*dx + (phi*2000)*inner(g,u_)*dx
p_form1 = -J1*inner(((J_a) ), grad(P1_))*dx + ( P1 - Pn1 )/dt * P1_ * dx + J1*g_a*P1_*ds(4) + J1*g_w*P1_*ds(1) + J1*g_w*P1_*ds(2)
p_form2 = -J1*inner(((J_w) ), grad(P2_))*dx + ( P2 - Pn2 )/dt * P2_ * dx + J1*g_w*P2_*ds(4) + J1*g_w*P2_*ds(1) + J1*g_w*P2_*ds(2)
p_form3 = phic_*(phi*J1 + phic + phi_w*J1 - J1)*dx
phi_form = p_*(p +pn -( gamma_a*R*T)/(1-b*gamma_a) + a*gamma_a**2 )*dx
psi_form = -inner(grad(P1),grad(psi_))*dx - psi*psi_*dx

F = mech_form + p_form1 + p_form2 + p_form3 + phi_form + psi_form
J = derivative(F, U, dU)


# In[315]:


problem = NonlinearVariationalProblem(F, U, bcs, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-4
prm['newton_solver']['relative_tolerance'] = 1E-8
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0


# In[333]:


t = 0
for n in range(1800):
    t += dt
    solver.solve()
    (u1, P11, P22, p1, phic1, psi1) = U.split()
    Un.assign(U)
    Pn1.copy(p11)
    Pn2.copy(P22)

s_a = (phic/J1)/(phi_w+(phic/J1))
s_a = project ( s_a, V.sub(1).collapse())
M5 = plot(s_a, title = '$Saturation_{CO2}$')
plt.colorbar(M5)
plt.show()
