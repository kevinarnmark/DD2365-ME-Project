import numpy as np
import time

from dolfin import *; from mshr import *

import dolfin.common.plotting as fenicsplot

from matplotlib import pyplot as plt

from os import mkdir, path
from shutil import rmtree

def simulate(nu, res, vis_mesh=False, vis_force=False):
  ###mkdir('results_ME')
  # Define domain and mesh
  ###

  # Define rectangular domain 
  L = 4
  H = 2

  # Define circle
  xc = 1.0
  yc = 0.5*H
  rc = 0.2

  # Parameters
  inflow_vel = 1.0
  rpm = 40

  # Define subdomains (for boundary conditions)
  class Left(SubDomain):
      def inside(self, x, on_boundary):
          return near(x[0], 0.0) 

  class Right(SubDomain):
      def inside(self, x, on_boundary):
          return near(x[0], L)

  class Lower(SubDomain):
      def inside(self, x, on_boundary):
          return near(x[1], 0.0)

  class Upper(SubDomain):
      def inside(self, x, on_boundary):
          return near(x[1], H)

  class Objects(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (not near(x[0], 0.0)) and (not near(x[0], L)) and (not near(x[1], 0.0)) and (not near(x[1], H))
        
  left = Left()
  right = Right()
  lower = Lower()
  upper = Upper()
  objects = Objects()

  # Generate mesh (examples with and without a hole in the mesh) 
  resolution = res
  rec = Rectangle(Point(0.0,0.0), Point(L,H))
  circle = Circle(Point(xc,yc),rc, resolution)
  domain = rec - circle
  domain.set_subdomain(1,circle)
  #mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), L*resolution, H*resolution)
  mesh = generate_mesh(domain, resolution)

  # create mesh function using domains already defined
  mf = MeshFunction("size_t", mesh, mesh.geometric_dimension(), mesh.domains())
  mf.set_all(0)
  objects.mark(mf, 1)


  # Local mesh refinement (specified by a cell marker)
  no_levels = 0
  for i in range(0,no_levels):
    cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
    for cell in cells(mesh):
      cell_marker[cell] = False
      p = cell.midpoint()
      if p.distance(Point(xc, yc)) < 1.0:
          cell_marker[cell] = True
    mesh = refine(mesh, cell_marker)

  # Define mesh functions (for boundary conditions)
  boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
  boundaries.set_all(0)
  left.mark(boundaries, 1)
  right.mark(boundaries, 2)
  lower.mark(boundaries, 3)
  upper.mark(boundaries, 4)
  objects.mark(boundaries, 5)


  # Define mesh functions (for force integration)
  obj_boundary = MeshFunction("bool", mesh, 0)
  obj_boundary.set_all(False)
  objects.mark(obj_boundary, True)

  coord_map = []
  idx_map = []

  for v in vertices(mesh):
    if obj_boundary[v]:
      coord_map.append((v.point()[0], v.point()[1]))
      idx_map.append(v.index())

  #print('coord_map: ', coord_map)
  #print(idx_map)
  #ss_coord_array = mesh.coordinates()[idx_map]
  #print('coord_array: ', ss_coord_array)



  # Calculating the Reynolds Number and printing information about the simulation
  re = (inflow_vel*rc*2)/nu
  print("Reynolds Number = ", re, "Mesh Resolution = 1/" + repr(res))

  if (vis_mesh):
    plt.figure()
    plot(mesh)
    plt.savefig(res_dir + '/mesh.png')


  ###
  # Define finite element approximation spaces
  ###

  # Generate finite element spaces (for velocity and pressure)
  V = VectorFunctionSpace(mesh, "Lagrange", 1)
  Q = FunctionSpace(mesh, "Lagrange", 1)

  # Define trial and test functions 
  u = TrialFunction(V)
  p = TrialFunction(Q)
  v = TestFunction(V)
  q = TestFunction(Q)


  ###
  # Define boundary conditions
  ###

  class DirichletBoundaryLower(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[1], 0.0)

  class DirichletBoundaryUpper(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[1], H)

  class DirichletBoundaryLeft(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[0], 0.0) 

  class DirichletBoundaryRight(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[0], L)

  class DirichletBoundaryObjects(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and (not near(x[0], 0.0)) and (not near(x[0], L)) and (not near(x[1], 0.0)) and (not near(x[1], H))

  dbc_lower = DirichletBoundaryLower()
  dbc_upper = DirichletBoundaryUpper()
  dbc_left = DirichletBoundaryLeft()
  dbc_right = DirichletBoundaryRight()
  dbc_objects = DirichletBoundaryObjects()

  # Examples of time dependent and stationary inflow conditions
  #uin = Expression('4.0*x[1]*(1-x[1])', element = V.sub(0).ufl_element())
  #uin = Expression('1.0 + 1.0*fabs(sin(t))', element = V.sub(0).ufl_element(), t=0.0)
  uin = inflow_vel
  bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
  bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)
  bcu_upp0 = DirichletBC(V.sub(0), 0.0, dbc_upper)
  bcu_upp1 = DirichletBC(V.sub(1), 0.0, dbc_upper)
  bcu_low0 = DirichletBC(V.sub(0), 0.0, dbc_lower)
  bcu_low1 = DirichletBC(V.sub(1), 0.0, dbc_lower)

  # Time step length 
  dt = 0.5*mesh.hmin() 

  omega = -np.pi/4.0 #-(rpm * 2 * np.pi * dt) / 60 
  o0 = cos(omega) #cos(-np.pi/4.0)
  o1 = sin(omega) #sin(-np.pi/4.0)
  mag = 0.0

  bc_exp0 = Expression('((xc + (x[0]-xc)*o0 - (x[1]-yc)*o1) - x[0])*mag', xc=xc, yc=yc, o0=o0, o1=o1, mag=mag, element = V.sub(0).ufl_element())
  bc_exp1 = Expression('((yc + (x[0]-xc)*o1 + (x[1]-yc)*o0) - x[1])*mag', xc=xc, yc=yc, o0=o0, o1=o1, mag=mag, element = V.sub(1).ufl_element())

  bcu_obj0 = DirichletBC(V.sub(0), bc_exp0, dbc_objects)
  bcu_obj1 = DirichletBC(V.sub(1), bc_exp1, dbc_objects)

  pin = Expression('5.0*fabs(sin(t))', element = Q.ufl_element(), t=0.0)
  pout = 0.0
  #bcp0 = DirichletBC(Q, pin, dbc_left) 
  bcp1 = DirichletBC(Q, pout, dbc_right)

  #bcu = [bcu_in0, bcu_in1, bcu_upp0, bcu_upp1, bcu_low0, bcu_low1, bcu_obj0, bcu_obj1]
  bcu = [bcu_in0, bcu_in1, bcu_upp1, bcu_low1, bcu_obj0, bcu_obj1]
  bcp = [bcp1]

  # Define measure for boundary integration  
  ds = Measure('dS', domain=mesh, subdomain_data=boundaries)


  ###
  # Define method parameters
  ###

  # Define iteration functions
  # (u0,p0) solution from previous time step
  # (u1,p1) linearized solution at present time step  
  u0 = Function(V)
  u1 = Function(V)
  p0 = Function(Q)
  p1 = Function(Q)

  # Set parameters for nonlinear and lienar solvers 
  num_nnlin_iter = 5 
  prec = "amg" if has_krylov_solver_preconditioner("amg") else "default" 


  ###
  # Define variational problem
  ###

  # Stabilization parameters
  h = CellDiameter(mesh)
  u_mag = sqrt(dot(u1,u1))
  d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0)))
  d2 = h*u_mag

  # Mean velocities for trapozoidal time stepping
  um = 0.5*(u + u0)
  um1 = 0.5*(u1 + u0)

  # Momentum variational equation on residual form
  Fu = inner((u - u0)/dt + grad(um)*um1, v)*dx - p1*div(v)*dx + nu*inner(grad(um), grad(v))*dx \
      + d1*inner((u - u0)/dt + grad(um)*um1 + grad(p1), grad(v)*um1)*dx + d2*div(um)*div(v)*dx 
  au = lhs(Fu)
  Lu = rhs(Fu)

  # Continuity variational equation on residual form
  Fp = d1*inner((u1 - u0)/dt + grad(um1)*um1 + grad(p), grad(q))*dx + div(um1)*q*dx 
  ap = lhs(Fp)
  Lp = rhs(Fp)


  ###
  # Compute force on boundary
  ###

  # Define the direction of the force to be computed 
  phi_x = 0.0
  phi_y = 1.0

  class PsiExpression(UserExpression):
    def __init__(self, phi_x, phi_y, coord_map, **kwargs):
      super().__init__(**kwargs)
      self.phi_x = phi_x
      self.phi_y = phi_y
      self.coord_map = coord_map

    def eval(self, values, x):
        i = 0
        for v in self.coord_map:
          if (near(x[0], v[0]) and near(x[1], v[1])):
            values[0] = self.phi_x
            values[1] = self.phi_y
            return
          i += 1
        values[0] = 0.0
        values[1] = 0.0

    def value_shape(self):
        return (2,)

  #psi_expression = Expression(("0.0","pow(x[0]-0.5,2.0) + pow(x[1]-1.0,2.0) - pow(0.2,2.0) < 1.e-5 ? 1. : 0."), element = V.ufl_element())
  #psi_expression = Expression(("near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_x : 0.","near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_y : 0."), xc=xc, yc=yc, rc=rc, phi_x=phi_x, phi_y=phi_y, element = V.ufl_element())
  Expression(phi_x, phi_y, coord_map, element=V.ufl_element())
  psi_expression = PsiExpression(phi_x, phi_y, coord_map, element=V.ufl_element())
  
  #test = project(psi_expression, V)
  #plot(mesh)


  psi = interpolate(psi_expression, V)
  plt.figure()
  plot(psi)
  plt.savefig(res_dir + '/userexpr' + '.png', dpi=300)

  Force = inner((u1 - u0)/dt + grad(um1)*um1, psi)*dx - p1*div(psi)*dx + nu*inner(grad(um1), grad(psi))*dx

  phi_x = 1.0
  phi_y = 0.0

  #psi_expression = Expression(("0.0","pow(x[0]-0.5,2.0) + pow(x[1]-1.0,2.0) - pow(0.2,2.0) < 1.e-5 ? 1. : 0."), element = V.ufl_element())
  #psi_expression_2 = Expression(("near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_x : 0.","near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_y : 0."), xc=xc, yc=yc, rc=rc, phi_x=phi_x, phi_y=phi_y, element = V.ufl_element())
  psi_expression_2 = PsiExpression(phi_x, phi_y, coord_map, element=V.ufl_element())
  
  psi_2 = interpolate(psi_expression_2, V)

  Force_2 = inner((u1 - u0)/dt + grad(um1)*um1, psi_2)*dx - p1*div(psi_2)*dx + nu*inner(grad(um1), grad(psi_2))*dx

  #plt.figure()
  #plot(psi, title="weight function psi")

  # Force normalization
  D = 2*rc
  #normalization = -2.0/D
  normalization = -2.0/(D*(uin**2))


  ###
  # Set plotting variables and open export files
  ###

  # Open files to export solution to Paraview
  file_u = File(res_dir + "/u.pvd")
  file_p = File(res_dir + "/p.pvd")

  # Set plot frequency
  plot_time = 0
  plot_freq = 10

  # Force computation data 
  force_array = np.array(0.0)
  force_array = np.delete(force_array, 0)
  force_array_2 = np.array(0.0)
  force_array_2 = np.delete(force_array_2, 0)
  time = np.array(0.0)
  time = np.delete(time, 0)
  start_sample_time = 1.0


  ###
  # Time stepping algorithm
  ###


  # Time stepping 
  T = 30
  t = dt

  while t < T + DOLFIN_EPS:

      #s = 'Time t = ' + repr(t) 
      #print(s)

      #pin.t = t
      #uin.t = t

      # Solve non-linear problem 
      k = 0
      while k < num_nnlin_iter: 
          
          # Assemble momentum matrix and vector 
          Au = assemble(au)
          bu = assemble(Lu)

          # Compute velocity solution 
          [bc.apply(Au, bu) for bc in bcu]
          [bc.apply(u1.vector()) for bc in bcu]
          solve(Au, u1.vector(), bu, "bicgstab", "default")

          # Assemble continuity matrix and vector
          Ap = assemble(ap) 
          bp = assemble(Lp)

          # Compute pressure solution 
          [bc.apply(Ap, bp) for bc in bcp]
          [bc.apply(p1.vector()) for bc in bcp]
          solve(Ap, p1.vector(), bp, "bicgstab", prec)

          # Compute force
          F = assemble(Force)
          F_2 = assemble(Force_2)
          if (t > start_sample_time):
            force_array = np.append(force_array, normalization*F)
            force_array_2 = np.append(force_array_2, normalization*F_2)
            time = np.append(time, t)

          k += 1

      if t > plot_time:
        file_u << u1
        file_p << p1
        plot_time += T/plot_freq
        s = 'Time t = ' + repr(t) 
        print(s)
    
      # Update time step
      u0.assign(u1)
      t += dt
      

  s = 'Time t = ' + repr(int(t)) #Set plotting variables and open export files
  #print(s)
      
  # Save solution to file
  #file_u << u1
  #file_p << p1

  # Plot solution
  plt.figure()
  plot(u1, title="Velocity, " + s + ", Re = " + repr(re))
  plt.savefig(res_dir + '/u' + repr(int(t)) + '.png', dpi=300)

  plt.figure()
  plot(p1, title="Pressure, " + s + ", Re = " + repr(re))
  plt.savefig(res_dir + '/p' + repr(int(t)) + '.png', dpi=300)

  plot_time += T/plot_freq
          

  if (vis_force):
    # Plot the lift force
    plt.figure()
    plt.title("Lift Force, " + s + ", Re = " + repr(re))
    plt.plot(time, force_array)

    # Plot the average of force_array
    lift_avg = np.array([sum(force_array) / len(force_array)]*len(time))
    plt.plot(time, lift_avg, color='red', label='avg = ' + "{:.3f}".format(lift_avg[0]))
    plt.legend()
    plt.savefig(res_dir + '/lift_force' + repr(int(t)) + '.png', dpi=300)

    # Plot the drag force
    plt.figure()
    plt.title("Drag Force, " + s + ", Re = " + repr(re))
    plt.plot(time, force_array_2)

    # Plot the average of force_array_2 during the last 5 seconds
    drag_avg = np.array([sum(force_array_2[int(len(force_array_2) * (T-5)/T):]) / len(force_array_2[int(len(force_array_2) * (T-5)/T):])]*len(time))
    plt.plot(time, drag_avg, color='red', label='avg = ' + "{:.3f}".format(drag_avg[0])) 
    plt.legend()
    plt.savefig(res_dir + '/drag_force' + repr(int(t)) + '.png', dpi=300)

  # Compute frequency when lift force is oscillating around 0
  # This is can also be used inside the time stepping loop to provide 
  # the strauhaul number for each plot. 
  # It has been moved here to only print at the end of each simulation.
  # Some parts of the code is therefore not used when placed here.
  start_t = -1
  end_t = 0
  if (t > start_sample_time):
    f_counter = 0
    before = force_array[0]
    for i in range(1, len(force_array)):
      if before > 0 and force_array[i] < 0: 
        if start_t < 0:
          start_t = time[i]
        f_counter += 1
        end_t = time[i]
      elif before < 0 and force_array[i] > 0:
        if start_t < 0:
          start_t = time[i]
        f_counter += 1
        end_t = time[i]
      before = force_array[i]
    if end_t - start_t > 0:
      freq = (f_counter / 2) / (end_t - start_t)
      print("Frequency of lift force oscillation: ", freq, " Hz")
      print("Strauhaul Number: ", (freq*2*rc)/uin, "\n")

  #!tar -czvf results_ME.tar.gz results_ME
  #files.download('results_ME.tar.gz')

res_dir = 'results_ME_1'
if (path.isdir(res_dir)):
  rmtree(res_dir)
  mkdir(res_dir)
else:
  mkdir(res_dir)

simulate(4e-3, 32, True, True)