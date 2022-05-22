from pickletools import unicodestring4
import numpy as np
import time as TIME
import sys
from dolfin import *; from mshr import *
import dolfin.common.plotting as fenicsplot
from matplotlib import pyplot as plt
from os import mkdir, path
from shutil import rmtree

def simulate(des, rpm, res):

    # Define circular domain 
    xc = 0.0
    yc = 0.0
    rc_domain = 60.0
    c_res = 16

    rc = 2.0

    if des == 0:
        # Define inner object circle
        rc = 2.0
        obj = Circle(Point(xc,yc), rc, c_res)
    elif des == 1:
        # Define rectangle object
        rc = 2.0/sqrt(2)
        obj = Rectangle(Point(-rc,-rc), Point(rc,rc))
    elif des == 2:
        # Define rectangle object
        rc = 2.0/sqrt(2)
    elif des == 3:
        # Define rectangle object
        rc = 2.0/sqrt(2)

    # Parameters
    uin = 10.0
    nu = 1.426e-5

    re = (uin*rc*2)/nu

    res_dir = "results_ALE_" + repr(des) + "_" + repr(rpm)
    if (path.isdir(res_dir)):
        rmtree(res_dir)
        mkdir(res_dir)
    else:
        mkdir(res_dir)

    # Define subdomains (for boundary conditions)
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and pow(x[0]-xc,2) + pow(x[1]-yc,2) > rc and x[0] <= xc

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and pow(x[0]-xc,2) + pow(x[1]-yc,2) > rc and x[0] > xc

    class Objects(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and pow(x[0]-xc,2) + pow(x[1]-yc,2) < rc_domain
        
    left = Left()
    right = Right()
    objects = Objects()
    
    #mesh = generate_mesh(Circle(Point(xc,yc), rc_domain, 2*c_res) - Circle(Point(xc,yc), rc, c_res), res) 
    mesh = generate_mesh(Circle(Point(xc,yc), rc_domain, 4*c_res) - obj, res) 
    h_unrefined = mesh.hmin()
    plt.figure()
    plot(mesh, linewidth=0.5)
    plt.savefig(res_dir + '/mesh.png', dpi=300)

    rc_refine = 1/3 * rc_domain

    # Local mesh refinement (specified by a cell marker)
    no_levels = 1
    for i in range(0,no_levels):
        cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
        for cell in cells(mesh):
            cell_marker[cell] = False
            p = cell.midpoint()
            if p.distance(Point(xc, yc)) < rc_refine:
            #if p[1] <= 0.3*(p[0] - csx) and p[1] >= -0.5*(p[0] + csx): # Does not work when rotating
                cell_marker[cell] = True
        mesh = refine(mesh, cell_marker)

    # Define mesh functions (for boundary conditions)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    right.mark(boundaries, 2)
    objects.mark(boundaries, 5)

    plt.figure()
    plot(mesh, linewidth=0.5)
    plt.savefig(res_dir + '/mesh_refined.png', dpi=300)

    # Generate finite element spaces (for velocity and pressure)
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    # Define trial and test functions 
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    dbc_left = Left()
    dbc_right = Right()
    dbc_objects = Objects()


    bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
    bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)

    slip0 = Expression('0.0', element = V.sub(0).ufl_element())

    bcu_obj0 = DirichletBC(V.sub(0), 0.0 if des==0 else slip0, dbc_objects)
    bcu_obj1 = DirichletBC(V.sub(1), 0.0, dbc_objects)

    pout = 0.0
    bcp1 = DirichletBC(Q, pout, dbc_right)

    bcu = [bcu_in0, bcu_in1, bcu_obj0, bcu_obj1]
    bcp = [bcp1]

    # Define measure for boundary integration  
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    # Define iteration functions
    # (u0,p0) solution from previous time step
    # (u1,p1) linearized solution at present time step  
    u0 = Function(V)
    u1 = Function(V)
    p0 = Function(Q)
    p1 = Function(Q)

    # Define mesh deformation w, mesh velocity = w/dt
    t = 0.0
    
    rot_vel_rc = rc * rpm * 2 * np.pi / 60
    print("Rotational velocity on rotor:", rot_vel_rc)
    rot_vel_d = rc_domain * rpm * 2 * np.pi / 60 # Velocity at the boundary of the domain
    rot_vel_ref = rc_refine * rpm * 2 * np.pi / 60 # Velocity at the outer edge of the refined mesh
    print("Rotational velocity on domain boundary:", rot_vel_d)

    #dt = mesh.hmin() / (rot_vel_d + uin) # Should be the magnitude of w and u0 (last time step)
    dt = min(h_unrefined / (rot_vel_d + uin), mesh.hmin() / (rot_vel_ref + uin)) # Smallest time-step depending on velocity compared to mesh size
    omega = -(rpm * 2 * np.pi * dt) / 60 #-pi/16.0
    o0 = cos(omega)
    o1 = sin(omega)
    #rot_vel = (rc * omega) / dt
    print("Debug Values", omega, o0, o1, rot_vel_rc)
    print("Debug Rotations:", (omega * (30/dt)) / (2*pi))
   
    w = Expression(('x[0]*o0-x[1]*o1 - x[0]','x[0]*o1+x[1]*o0 - x[1]'), rc=rc, dt=dt, o0=o0, o1=o1, xc=xc, yc=yc, element = V.ufl_element())

    # Set parameters for nonlinear and lienar solvers 
    num_nnlin_iter = 5 
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default" 


    # Define variational problem

    # Stabilization parameters
    h = CellDiameter(mesh)
    u_mag = sqrt(dot(u1,u1))
    d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0)))
    d2 = h*u_mag

    # Mean velocities for trapozoidal time stepping
    um = 0.5*(u + u0)
    um1 = 0.5*(u1 + u0)

    # Momentum variational equation on residual form
    Fu = inner((u - u0)/dt + grad(um)*(um1-w/dt), v)*dx - p1*div(v)*dx + nu*inner(grad(um), grad(v))*dx \
        + d1*inner((u - u0)/dt + grad(um)*(um1-w/dt) + grad(p1), grad(v)*(um1-w/dt))*dx + d2*div(um)*div(v)*dx 
    au = lhs(Fu)
    Lu = rhs(Fu)

    # Continuity variational equation on residual form
    Fp = d1*inner((u1 - u0)/dt + grad(um1)*(um1-w/dt) + grad(p), grad(q))*dx + div(um1)*q*dx 
    ap = lhs(Fp)
    Lp = rhs(Fp)

    # Define the direction of the force to be computed 
    phi_x = 0.0
    phi_y = 1.0

    #psi_expression = Expression(("0.0","pow(x[0]-0.5,2.0) + pow(x[1]-1.0,2.0) - pow(0.2,2.0) < 1.e-5 ? 1. : 0."), element = V.ufl_element())
    
    psi_expression = Expression(("0.0","0.0"), element = V.ufl_element())

    #psi expression for circle:
    if des == 0:
        psi_expression = Expression(("near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_x : 0.","near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_y : 0."), xc=xc, yc=yc, rc=rc, phi_x=phi_x, phi_y=phi_y, element = V.ufl_element())
    elif des == 1:
        # Psi expression for square
        psi_expression = Expression(("(near(abs(x[0]), abs(rc)) && near(abs(x[1]), abs(rc))) ? phi_x : 0.","near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_y : 0."), xc=xc, yc=yc, rc=rc, phi_x=phi_x, phi_y=phi_y, element = V.ufl_element())
    
    psi_expression = Expression(("near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_x : 0.","near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_y : 0."), xc=xc, yc=yc, rc=rc, phi_x=phi_x, phi_y=phi_y, element = V.ufl_element())
    psi = interpolate(psi_expression, V)
    #psi = DirichletBC(V.sub(0), phi_x, dbc_objects)

    Force = inner((u1 - u0)/dt + grad(um1)*um1, psi)*dx - p1*div(psi)*dx + nu*inner(grad(um1), grad(psi))*dx

    #plt.figure()
    #plot(psi, title="weight function psi")

    # Force normalization
    D = 2*rc
    normalization = -2.0/D

    # Open files to export solution to Paraview
    file_u = File(res_dir + "/u.pvd")
    file_p = File(res_dir + "/p.pvd")

    # Force computation data 
    force_array = np.array(0.0)
    force_array = np.delete(force_array, 0)
    time = np.array(0.0)
    time = np.delete(time, 0)
    start_sample_time = 1.0


    # Time stepping 
    T = 30
    t = dt

    # Set plot frequency
    frame_rate = 2
    plot_time = 0
    plot_freq = T*frame_rate

    start_time = TIME.time()

    while t < T + DOLFIN_EPS:

        ALE.move(mesh, w) # Update mesh deformation
        
        # Reset boundary conditions since ALE.move moves both mesh and bc:
        #dbc_left = DirichletBoundaryLeft()
        #dbc_right = DirichletBoundaryRight()
        #dbc_objects = DirichletBoundaryObjects()

        bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
        bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)

        bcu_obj0 = DirichletBC(V.sub(0), 0.0, dbc_objects)
        bcu_obj1 = DirichletBC(V.sub(1), 0.0, dbc_objects)

        pout = 0.0
        bcp1 = DirichletBC(Q, pout, dbc_right)

        bcu = [bcu_in0, bcu_in1, bcu_obj0, bcu_obj1]
        bcp = [bcp1]
        
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
            if (t > start_sample_time):
                force_array = np.append(force_array, normalization*F)
                time = np.append(time, t)

            k += 1

        if t > plot_time:     
            
            s = 'Time t = ' + repr(t) 
            print(s)
        
            # Save solution to file
            file_u << u1
            file_p << p1

            
            # Plot solution
            plt.figure()
            plot(u1, title="Velocity")
            plt.savefig(res_dir + "/u" + repr(t) + ".png", dpi=300)
            plt.close()
            """
            plt.figure()
            plot(p1, title="Pressure")
            plt.savefig(res_dir + "/p" + repr(t) + ".png", dpi=300)

            plt.figure()
            plot(mesh, title="Mesh")
            plt.savefig(res_dir + "/mesh"  + repr(t) + ".png", dpi=300)

            plt.figure()
            plt.title("Force")
            plt.plot(time, force_array)
            plt.savefig(res_dir + "/force"  + repr(t) + ".png", dpi=300)
            """

            plot_time += T/plot_freq



        # Update time step
        u0.assign(u1)
        t += dt



    # Plot solution
    plt.figure()
    plot(u1, title="Velocity, " + s + ", Re = " + repr(int(re)))
    plt.savefig(res_dir + "/u" + repr(int(t)) + '.png', dpi=300)

    plt.figure()
    plot(p1, title="Pressure, " + s + ", Re = " + repr(int(re)))
    plt.savefig(res_dir + "/p" + repr(int(t)) + '.png', dpi=300)

    # Plot the lift force
    plt.figure()
    plt.title("Lift Force, " + s + ", Re = " + repr(int(re)))
    plt.plot(time, force_array)

    # Plot the average of force_array
    lift_avg = np.array([sum(force_array) / len(force_array)]*len(time))
    plt.plot(time, lift_avg, color='red', label='avg = ' + "{:.3f}".format(lift_avg[0]))
    plt.legend()
    plt.savefig(res_dir + "/lift_force" + repr(int(t)) + '.png', dpi=300)

    plt.close('all')

    print("Simulation Time:", TIME.time() - start_time)


args = sys.argv
print("Run in parallel using GNU Parallel.\n Example: parallel python3 ALE_Magnus.py ::: 0 1 ::: 40 80 120.\n Runs simulation for design 0 and 1 with rpm 40, 80 & 120, total 6 sims")
print("Starting simulation for", int(args[2]), "rpm")
simulate(int(args[1]), int(args[2]), 32)