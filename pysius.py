
# coding: utf-8

# # Pysius - a Python solver for Blasius equation

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt


# Let $f(\eta)$ be the non-dimensional stream function, with $\eta$ being the similarity variable. Blasius equation reads:
# 
# $$f''' + \frac{1}{2}ff'' = 0$$
# 
# This is a third-order ODE, that can be thus cast to a system of 3 first-order ODEs:
# 
# $$\begin{cases}
# f' = u\\
# u'  = \xi\\
# \xi' = -\frac{1}{2}f\xi
# \end{cases}$$
# 
# the first two rows representing de-facto the definitions of the additional variables $f$ and $\xi$. Boundary conditions are:
# 
# $$f(0)=0$$
# $$u(0)=0$$
# $$\lim_{\eta \to \infty}u(\eta)=\int_0^\infty \xi d\eta = 1$$
# 
# A complete set of initial conditions is not available (the value of $\xi$ at $\eta=0$ is not provided). This makes it impossible to solve the equation with an $\eta$-marching scheme (meaning, starting from initial conditions towards higher values of eta), hence they need to be solved nonlinearly. Also, the third condition can be either written as a condition on $u$ or an integral condition on $\xi$; the first option will be used in this script (although the same result can be obtained with the latter).

# The theoretical domain would be $\eta\in[0,\infty)$; a discretisation is needed. Here, $\infty$ is approximated with a high enough value `eta_max`, while the parameter `N` indicates  the number of nodes on which the function is evaluated (including extremes, so including $0$ and `eta_max`. Finally `deta` is the spacing between nodes.
# 
# _Hint: try starting from a low number of nodes and a low value for_ `eta_max` _and gradually increase them. Do this until no substantial change in the final solution is observed._

# In[2]:


eta_max = 20 # extreme of domain
N = 1000 # number of nodes

deta = eta_max / (N-1)


# Next up, it has to be decided how to store the unknowns in memory. There are three variables, $f$, $u$ and $\xi$, and $N$ nodes - which adds up to $3N$ unknowns (corresponding to the value of each variable on each node). All of them are stored in a one-dimensional array $y$ of size $3N$:

# In[3]:


y = np.zeros(3*N)


# It is conventionally decided that the first N elements of `y` represent $\xi$ at each of the nodes, namely:
# ```python
# xi = y[0:N] # where xi[n-1] represents xi at the n-th node
# ```
# notice that, when using Python's slicing `0:N`, the first extreme ($0$) is inclusive, the second ($N$) is not. The other variables just follow:
# ```python
# u = y[N:(2*N)]
# f = y[(2*N):(3*N)]
# ```

# Using a forward-Euler discretisation, the equation can be evaluated at each node $n$ as:
# 
# $$
# \left(
# \begin{bmatrix}
# \xi_{n+1}\\
# u_{n+1}\\
# f_{n+1}
# \end{bmatrix} - 
# \begin{bmatrix}
# \xi_{n}\\
# u_{n}\\
# f_{n}
# \end{bmatrix} 
# \right) \frac{1}{\Delta \eta} +
# \begin{bmatrix}
# \frac{1}{2}\xi_{n}f_n\\
# -\xi_{n}\\
# -u_{n}
# \end{bmatrix} = 0
# $$
# 
# Notice that such equation cannot be written for $n=N$ (the last cell), as it would require informations from the unexistent cell $N+1$. Therefore, $3(N-1)$ equations can be written on all the other nodes; the remaining 3 equations are provided by boundary conditions, so that the number of equations matches the one of unknowns.
# 
# This can be written as a whole as:
# 
# $$
# Ay + b(y) = 0
# $$
# 
# where $A$ is a matrix, vector $b$ is a function of y. The whole left hand side is thus a vector, where the arrangement of rows is arbitrary; here it is performed so that:
# - the first $N-1$ rows represent the discretised Blasius equation for $\xi$
# - the next row represents the boundary contition for $u(0) = 0$
# - the next $N-1$ rows represent the discretised Blasius equation for $u$
# - the next row represents the boundary condition for $u(\eta\to\infty)$
# - the next row is the boundary condition for $f(0)$
# - the final $N-1$ rows represent the discretised Blasius equation for $f$

# In[4]:


# Keep in mind that when slicing or using "range", the first bound is inclusive
# while the second is not!
# In this way, for instance, range(1,11) produces an array of numbers from 1 to 10,
# which has size 11-1 = 10.

A = np.zeros((3*N, 3*N))

# Blasius equation for xi (derivative of xi)
for ii in range(N-1):
    A[ii,ii] = -1/deta
    A[ii,ii+1] = 1/deta

# boundary condition for u(0)
A[N-1, N] = 1

# Blasius equation for u (derivative of u)
for ii in range(N,2*N-1):
    A[ii,ii] = -1/deta
    A[ii,ii+1] = 1/deta

# boundary conditions for u at etamax ~ infinity
A[2*N-1,2*N-1] = 1

# boundary condition for f(0)
A[2*N,2*N] = 1

# Blasius equation for f (derivative of f)
for ii in range(2*N+1, 3*N):
    A[ii,ii-1] = -1/deta
    A[ii,ii] = 1/deta


# In[5]:


# now, a function returning vector b is defined

def b(y):
    
    result = np.zeros(3*N)
    
    # Blasius equation for xi (derivative of xi)
    for ii in range(N-1):
        result[ii] = y[ii]*y[2*N+ii]/2
    
    # Blasius equation for u (derivative of u)
    for ii in range(N,2*N-1):
        result[ii] = - y[ii-N]
    
    # boundary conditions for u at etamax ~ infinity
    result[2*N-1] = -1
    
    # Blasius equation for f (derivative of f)
    for ii in range(2*N+1, 3*N):
        result[ii] = -y[ii-1-N]
    
    return result


# In[6]:


# Jacobi's matrix for b; this will come in handy later

def db(y):
    
    result = np.zeros((3*N,3*N))
    
    # Blasius equation for xi (derivative of xi)
    for ii in range(N-1):
        result[ii,2*N+ii] = y[ii]/2
        result[ii,ii] = y[2*N+ii]/2
        
    # Blasius equation for u (derivative of u)
    for ii in range(N,2*N-1):
        result[ii,ii-N] = - 1

    # Blasius equation for f (derivative of f)
    for ii in range(2*N+1, 3*N):
        result[ii,ii-1-N] = -1
        
    return result


# We now have a non-linear equation in the form:
# 
# $$ F(y) = Ay + b(y) = 0$$
# 
# this can be solved with a Newton-Raphson scheme starting from an initial guess `y0`:
# 
# $$ y^{\,k+1} = y^{\,k} - J^{-1}(y^{\,k})\, F(y^{\,k}) $$
# 
# where $J$ is the Jacobi's matrix of $F$, while $k$ denotes the iteration of the algorithm. Iterations are stopped when a maximum of iterations `KMAX` is reached or when the infinity norm of the vector of the change between iterations $||y^{\,k+1} - y^{\,k}||_\infty$ is below a threshold value `TOL`.

# In[7]:


# define F and the jacobian J
F = lambda yy : A.dot(yy) + b(yy)
J = lambda yy : A + db(yy)


# In[8]:


# initial condition (linear profiles which respect B.C.)
y0 = np.zeros(3*N)
y0[0:N] = 0.5*np.linspace(1.0,0.0,num=N)*deta
y0[N:2*N] = np.linspace(0,1,num=N)
y0[2*N:3*N] = np.linspace(0,1,num=N)


# In[9]:


KMAX = 100 # maximum number of iterations
TOL = 1e-10 # tolerance

exit4it = True
for ii in range(KMAX):
    dy = - np.linalg.solve(J(y0),F(y0))
    y = y0 + dy
    y0 = y
    print('Iteration {}, maximum absolute value of residual: {}'.format(ii+1,max(abs(F(y)))))
    if max(abs(dy)) < TOL:
        exit4it = False
        print()
        print('Algorithm terminated due to satisfactory residual (number of iterations: {}).'.format(ii+1))
        break
if exit4it:
    print()
    print('Warning: algorithm was terminated for too many iterations;')
    print('converngence might have not been achieved.')


# Now, the solution is extracted and plotted:

# In[10]:


eta = np.linspace(0,eta_max,num=N)
xi = y[0:N]
u = y[N:(2*N)]
f = y[(2*N):(3*N)]


# In[11]:


fig, ax = plt.subplots()
ax.plot(eta,f)
ax.plot(eta,u)
ax.plot(eta,xi)
ax.set_xlabel('$\eta$')
ax.set_xlim([0,10])
ax.set_ylim([-0.1,1.3])
ax.grid()
ax.legend(['f', 'u', 'xi'])

# # Postprocessing

# Now, the streamlines associated to the Blasius solution are plotted in a $x,z$ plane; these correspond to the isocontours of the dimensional stream function $\psi$. It is thus necessary to first find the values of $f$ in real space (as a function of $x$ and $y$). To do so, a mesh is created; then, the value of $\eta$ corresponding to each couple $(x,y)$ is calculated:
# 
# $$ \eta = y\sqrt{\frac{u_\infty}{\nu x}} $$
# 
# Knowing $\eta$, it is possible to find the value of $f$ for each $(x,y)$ by interpolation of the Blasius solution; hence, it is also possible to calculate the dimensional stream function $\psi$:
# 
# $$ \psi = f\sqrt{\nu u_\infty x} $$

# In[12]:


# first off: define u_inf and nu
# unitary values are arbitrarily chosen
u_inf = 1
nu = 1


# In[13]:


# function that returns eta from x and y
get_eta = lambda x,y : y / math.sqrt(nu*x/u_inf)


# When creating a mesh, one needs to be careful: $x=0$ is __outside__ of the domain for Blasius boundary layer (also, the solution is discontinuous there). Moreover, a value $x=0$ would make $\eta$ undefined (see its definition).

# In[14]:


# mesh settings
x_min = 0.05; x_max = 1 # bounds for x
y_max = 1; # bounds for y (lower bound is assumed to be 0)
nox = 30; noy = 30 # number of points in each direction

# generate mesh
if get_eta(x_min, y_max) > eta_max:
    print('Warning: requested mesh exceeds the calculated values of eta.')
x, y = np.meshgrid(np.linspace(x_min,x_max,num=nox), np.linspace(0,y_max,num=noy))


# In[15]:


# calculate stream function associated to each x, y
psi = np.empty_like(x)
for ii in range(noy):
    for jj in range(nox):
        current_eta = get_eta(x[ii,jj], y[ii,jj])
        psi[ii,jj] = np.interp(current_eta, eta, f) * math.sqrt(nu*u_inf*x[ii,jj])


# In[16]:


fig, ax = plt.subplots()
ax.contour(x,y,psi,levels=[0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16])
ax.set_title('Streamlines')
ax.set_xlabel('x')
ax.set_ylabel('y')


# And finally, the non-dimensional velocity field can be found as:
# 
# $$ u_x = u_\infty \, f' $$
# 
# $$ u_y = \sqrt{\frac{\nu u_\infty}{x}} \, \left(\eta\, f' -f \right) $$
# 
# _Notice that this leads to an infinite value of $u_y$ in the limit for $x \to 0$._

# In[17]:


# mesh is here redefined to have better looking graphs
nox = math.floor(nox/4) # number of points in x is divided by 4
noy = math.floor(noy/1.5) # number of points in y is divided by 1.5
x, y = np.meshgrid(np.linspace(x_min,x_max,num=nox), np.linspace(0,y_max,num=noy))

# calculate velocity field associated to each x, y
ux = np.empty_like(x)
uy = np.empty_like(x)
for ii in range(noy):
    for jj in range(nox):
        current_eta = get_eta(x[ii,jj], y[ii,jj])
        ux[ii,jj] = np.interp(current_eta, eta, u) * u_inf
        uy[ii,jj] = (current_eta*np.interp(current_eta, eta, u) - np.interp(current_eta, eta, f)) * math.sqrt(nu*u_inf/x[ii,jj])


# In[18]:


fig, ax = plt.subplots()
ax.quiver(x,y,ux,uy)
ax.set_title('Velocity field')
ax.set_xlabel('x')
ax.set_ylabel('y')


# Again, the high values of $u_y$ towards $x \to 0$ are a peculiarity of Blasius' solution; this is a direct consequence of the similarity. The $u_x$ velocity component is assumed to be uniform and equal to $u_\infty$ for $x\leq0$ (as the external solution is assumed to have such shape); in other words, for $x=0$ and $\forall y$, $u_x = u_\infty$. For $x>0$, on the other hand, $u_x$ is given by a self similar solution for the boundary layer; the only way to satisfy both these conditions is to have a zero-thickness of the boundary layer in $x=0$, which also makes sense on a physical standpoint. As $x=0$ is approached, the y-profile of the self similar solution for $u_x$ becomes increasingly flat with an almost constant value of $u_\infty$, as a consequence of the thinner and thinner boundary layer. This can be clearly observed below:

# In[19]:


x, y = np.meshgrid(np.linspace(0.01,x_max,num=nox), np.linspace(0,y_max,num=noy))
ux = np.empty_like(x)
uy = np.empty_like(x)
for ii in range(noy):
    for jj in range(nox):
        current_eta = get_eta(x[ii,jj], y[ii,jj])
        ux[ii,jj] = np.interp(current_eta, eta, u) * u_inf
        uy[ii,jj] = (current_eta*np.interp(current_eta, eta, u) - np.interp(current_eta, eta, f)) * math.sqrt(nu*u_inf/x[ii,jj])

fig, ax = plt.subplots()
ax.quiver(x,y,ux,0)
ax.set_title('x-component of velocity field')
ax.set_xlabel('x')
ax.set_ylabel('y')


# However, the fact that the boundary layer thickness (from now on indicated with $\delta(x)$) tends to 0 for $x \to 0$ also means that $\eta$ becomes singular there, as it can be written as:
# 
# $$ \eta = \frac{y}{\delta(x)} $$
# 
# Similarly, also the vertical velocity component $u_y$ becomes singular.
# 
# Another issue with the vertical velocity component is that, for any $x$ and for $y \to \infty$, non-zero values are observed (actually, $u_y$ becomes larger and larger moving away from the wall, until an asymptotic value is found); this does not match the imposed external, irrotational flow, which is assumed to be:
# 
# $$\mathbf{U}_{ext} = u_\infty\hat{\mathbf{x}} + 0\,\hat{\mathbf{y}}$$
# 
# and hence has a null vertical component. Be aware that, despite the mismatch, Blasius solution is mathematically correct; a possible explanation for the incongruence resides in the non-physical choice of external velocity $\mathbf{U}_{ext}$. This has been calculated by solving Euler's equations with non penetration boundary conditions applied at the plate; meaning, the presence of the boundary layer is not taken into account. In fact, the external solution must phisically be aware of the boundary layer; this can be done numerically by iteratively modifying the external solution with the associated boundary layer one, then calculating a new boundary layer solution. In other words:
# - calculate boundary layer solution associated to external solution;
# - update external solution (which is, calculate a new one) so that it accounts for the presence of the boundary layer previously calculated;
# - repeat until convergence (if convergence is ever obtained!).

# In[ ]:

plt.show()


