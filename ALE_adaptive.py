from pickletools import unicodestring4
import numpy as np
import time as TIME
import sys
from dolfin import *; from mshr import *
import dolfin.common.plotting as fenicsplot
from matplotlib import pyplot as plt
from os import mkdir, path
import signal
import time

shut_down = False

def handler(signum, frame):
    global shut_down
    res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
    if res == 'y':
        shut_down = True
 
signal.signal(signal.SIGINT, handler)

args = sys.argv
print("Starting simulation for design", int(args[1]), "with" ,int(args[2]), "rpm", "with resolution", int(args[3]), "\n")

print("Run in parallel using GNU Parallel.\n Example: parallel python3 ALE_Magnus.py ::: 0 1 ::: 40 80 120.\n Runs simulation for design 0 and 1 with rpm 40, 80 & 120, total 6 sims", "\n")

print("Input Ctrl-c + y to safely exit the simulation, so it can be continued from a checkpoint", "\n")

des, rpm, res = int(args[1]), int(args[2]), int(args[3])

# Parameters
uin = 10.0
nu = 1.426e-5

# Define circular domain 
xc = 0.0
yc = 0.0
rc_domain = 60.0
c_res = 16

rc = 2.0

# Define object
if des == 0:
    # Define inner object circle
    obj = Circle(Point(xc,yc), rc, c_res)
elif des == 1:
    # Define rectangle object
    rc_coord = rc / sqrt(2)
    obj = Rectangle(Point(-rc_coord,-rc_coord), Point(rc_coord,rc_coord))
elif des == 2:
    # Define circle with 4 fins object
    rc_coord = rc / sqrt(2)
    obj = Circle(Point(xc,yc), rc, c_res) + Rectangle(Point(-rc_coord/4,-2*rc_coord), Point(rc_coord/4,2*rc_coord)) \
            + Rectangle(Point(-2*rc_coord,-rc_coord/4), Point(2*rc_coord,rc_coord/4))
elif des == 3:
    # Define debug object
    rc_coord = rc / sqrt(2)
    obj = Rectangle(Point(-rc_coord/2,-2*rc_coord), Point(rc_coord/2,2*rc_coord))

re = (uin*rc*2)/nu

print("RE =", repr(re))

saved_state = False
ss = None # 0 - time, 1 - plot time, 2 - plot figure time, 3 - checkpoint number

res_dir = "0_final_results_" + repr(des) + "_" + repr(rpm) + "_" + repr(res) + "_" + repr(int(re))
if (path.isdir(res_dir)):
    saved_state = True
    ss = np.load(res_dir + "/saved_state.npy")
    force_cp = np.load(res_dir + "/force_checkpoint.npy")
    ss_time = ss[0]
    ss_plt = ss[1]
    ss_plt_f = ss[2]
    ss_nr = ss[3] + 1

    ss_u1 = res_dir + "/u.xml"
    ss_p1 = res_dir + "/p.xml"

    print("Restoring from checkpoint, time:", f"{ss_time:.1f}")
else:
    ss_nr = 0
    mkdir(res_dir)



# Define subdomains (for boundary conditions)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and pow(x[0]-xc,2) + pow(x[1]-yc,2) > pow(1.1*rc,2) and x[0] <= xc

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and pow(x[0]-xc,2) + pow(x[1]-yc,2) > pow(1.1*rc,2) and x[0] > xc

class Objects(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and pow(x[0]-xc,2) + pow(x[1]-yc,2) < pow(0.9*rc_domain,2)
    
left = Left()
right = Right()
objects = Objects()

mesh = generate_mesh(Circle(Point(xc,yc), rc_domain, 4*c_res) - obj, res) 

plt.figure()
plot(mesh, linewidth=0.5)
plt.savefig(res_dir + '/mesh.png', dpi=300)

rc_refine = 1/3 * rc_domain
# Local mesh refinement (specified by a cell marker)
no_levels = 1
for _ in range(0,no_levels):
    cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
    for cell in cells(mesh):
        cell_marker[cell] = False
        p = cell.midpoint()
        if p.distance(Point(xc, yc)) < rc_refine:
        #if p[1] <= 0.3*(p[0] - csx) and p[1] >= -0.5*(p[0] + csx): # Does not work when rotating
            cell_marker[cell] = True
    mesh = refine(mesh, cell_marker)

np.save(res_dir + "/num_cells_" + str(mesh.num_cells()) + ".npy", np.array([mesh.num_cells()]))

plt.figure()
plot(mesh, linewidth=0.5)
plt.savefig(res_dir + '/mesh_refined.png', dpi=300)

# Define mesh functions (for boundary conditions)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
objects.mark(boundaries, 5)

# Define mesh function (for weight function psi)
obj_boundary = MeshFunction("bool", mesh, 0)
obj_boundary.set_all(False)
objects.mark(obj_boundary, True)

coord_map = {} # For the expression for psi to set 

for v in vertices(mesh):
    if obj_boundary[v]:
        coord_map[(v.point()[0], v.point()[1])] = True
        

# Generate finite element spaces (for velocity and pressure)
V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions 
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Define mesh deformation w, mesh velocity = w/dt
t = 0.0

rot_vel_rc = rc * rpm * 2 * np.pi / 60
print("Rotational velocity on rotor:", rot_vel_rc)
rot_vel_d = rc_domain * rpm * 2 * np.pi / 60 # Velocity at the boundary of the domain
#rot_vel_ref = rc_refine * rpm * 2 * np.pi / 60 # Velocity at the outer edge of the refined mesh
#rot_vel_obj = (rc + domain_h_unref) * rpm * 2 * np.pi / 60 # Velocity at the object boundary of the refined mesh
print("Rotational velocity on domain boundary:", rot_vel_d)

#dt = Constant(min(domain_h_unref / (rot_vel_d + uin), domain_hmin / (rot_vel_ref + uin), obj_hmin / (rot_vel_obj + uin))) # Smallest time-step depending on velocity compared to mesh size
dt = Constant(mesh.hmin() / (rot_vel_d + uin)) # Smallest possible timestep that could be needed for CFL condition during the first iteration
omega = -(rpm * 2 * np.pi * dt) / 60 
o0 = Constant(cos(omega))
o1 = Constant(sin(omega))

w = Expression(('x[0]*o0-x[1]*o1 - x[0]','x[0]*o1+x[1]*o0 - x[1]'), rc=rc, dt=dt, o0=o0, o1=o1, xc=xc, yc=yc, element = V.ufl_element())
w_bc = Expression(('(x[0]*o0-x[1]*o1 - x[0]) / dt','(x[0]*o1+x[1]*o0 - x[1]) / dt'), rc=rc, dt=dt, o0=o0, o1=o1, xc=xc, yc=yc, element = V.ufl_element())


# Defining boundary conditions:
dbc_left = Left()
dbc_right = Right()
dbc_objects = Objects()

bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)

bcu_obj = DirichletBC(V, w_bc, dbc_objects)

pout = 0.0
bcp1 = DirichletBC(Q, pout, dbc_right)

bcu = [bcu_in0, bcu_in1, bcu_obj]
bcp = [bcp1]


# Plot boundary conditions
bcu_f = Function(V)
bcp_f = Function(Q)

[bc.apply(bcu_f.vector()) for bc in bcu]
plt.figure()
plot(bcu_f)
plt.savefig(res_dir + '/bcu' + '.png', dpi=300)

bcp_f.vector()[:] = -1.0
[bc.apply(bcp_f.vector()) for bc in bcp]
plt.figure()
plot(bcp_f)
plt.savefig(res_dir + '/bcp' + '.png', dpi=300)


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

# Time scalar field
w0 = interpolate(w_bc, V)
w_mag = sqrt(dot(w0,w0))
dt_sf = CellDiameter(mesh) / sqrt(dot(u0 - w0,u0 - w0))# (u_mag + w_mag)
DG0 = FunctionSpace(mesh,"DG",0) # FunctionSpace to project time scalar field to
#print("Debug magnitude", project(sqrt(dot(u_mag,w_mag)), DG0).vector().min())

class PsiExpression(UserExpression):
    def __init__(self, phi_x, phi_y, coord_map, **kwargs):
        super().__init__(**kwargs)
        self.phi_x = phi_x
        self.phi_y = phi_y
        self.coord_map = coord_map

    def eval(self, values, x):
        if (x[0],x[1]) in coord_map:
            values[0] = self.phi_x
            values[1] = self.phi_y
        else:    
            values[0] = 0.0
            values[1] = 0.0

    def value_shape(self):
        return (2,)

psi_l_expression = PsiExpression(0.0, 1.0, coord_map, element=V.ufl_element())
psi_d_expression = PsiExpression(1.0, 0.0, coord_map, element=V.ufl_element())
psi_l = interpolate(psi_l_expression, V)
psi_d = interpolate(psi_d_expression, V)

plt.figure()
plot(psi_l)
plt.savefig(res_dir + '/psi_expression' + '.png', dpi=300)

Force_l = inner((u1 - u0)/dt + grad(um1)*um1, psi_l)*dx - p1*div(psi_l)*dx + nu*inner(grad(um1), grad(psi_l))*dx
Force_d = inner((u1 - u0)/dt + grad(um1)*um1, psi_d)*dx - p1*div(psi_d)*dx + nu*inner(grad(um1), grad(psi_d))*dx

# Force normalization
D = 2*rc
normalization = -2.0/(uin**2 * D * 1.2466) # Air at 10 degrees celsius is 1.2466kg/m**3

# Open files to export solution to Paraview
if saved_state:
    file_u = File(res_dir + "/u_" + repr(int(ss_nr)) + ".pvd")
    file_p = File(res_dir + "/p_" + repr(int(ss_nr)) + ".pvd")
else:
    file_u = File(res_dir + "/u.pvd")
    file_p = File(res_dir + "/p.pvd")

# Solutions for saved state
file_u_ss = File(res_dir + "/u.xml")
file_p_ss = File(res_dir + "/p.xml")

# Force computation data 
lift_coeff_array = np.array(0.0)
lift_coeff_array = np.delete(lift_coeff_array, 0)
force_l_array = np.array(0.0)
force_l_array = np.delete(force_l_array, 0)
drag_coeff_array = np.array(0.0)
drag_coeff_array = np.delete(drag_coeff_array, 0)
force_d_array = np.array(0.0)
force_d_array = np.delete(force_d_array, 0)
time = np.array(0.0)
time = np.delete(time, 0)
start_sample_time = 1.0

if (saved_state):
    lift_coeff_array = force_cp[0]
    force_l_array = force_cp[1]
    drag_coeff_array = force_cp[2]
    force_d_array = force_cp[3]
    time = force_cp[4]

# Time stepping 
T = 30
t = dt((5,7))

# Set plot frequency
frame_rate = 2
plot_time = 0
plot_time_fig = 0
plot_freq = T*frame_rate
plot_freq_fig = T*0.5

# Assign data from saved state
if (saved_state):
    ss_rot = ((ss_time) / dt((5,7))) * omega((5,7))
    ss_w = Expression(('x[0]*o0-x[1]*o1 - x[0]','x[0]*o1+x[1]*o0 - x[1]'), rc=rc, o0=cos(ss_rot), o1=sin(ss_rot), xc=xc, yc=yc, element = V.ufl_element())
    ALE.move(mesh, ss_w)
    ss_u = Function(V, ss_u1)
    ss_p = Function(Q, ss_p1)
    u0.assign(ss_u)
    u1.assign(ss_u)
    p1.assign(ss_p)
    t = ss_time + dt((5,7))
    plot_time = ss_plt
    plot_time_fig = ss_plt_f



start_time = TIME.time()
first = True

while t < T + DOLFIN_EPS and not shut_down:

    ALE.move(mesh, w) # Update mesh deformation / move mesh
    
    # Update boundary conditions:
    bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
    bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)
    bcu_obj = DirichletBC(V, w_bc, dbc_objects)

    pout = 0.0
    bcp1 = DirichletBC(Q, pout, dbc_right)

    bcu = [bcu_in0, bcu_in1, bcu_obj]
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
        F_l = assemble(Force_l)
        F_d = assemble(Force_d)
        if (t > start_sample_time):
            lift_coeff_array = np.append(lift_coeff_array, normalization*F_l)
            force_l_array = np.append(force_l_array, F_l)
            drag_coeff_array = np.append(drag_coeff_array, normalization*F_d)
            force_d_array = np.append(force_d_array, F_d)
            time = np.append(time, t)

        k += 1

    if t > plot_time:     
        
        s = 'Time t = ' + repr(t) 
        print(s)
    
        # Save solution to file
        file_u << u1
        file_p << p1
        # Save solutions to be used as checkpoints
        file_u_ss << u1
        file_p_ss << p1

        plot_time += T/plot_freq
        # Save for use as checkpoint
        np.save(res_dir + "/saved_state.npy", np.array([t, plot_time, plot_time_fig, ss_nr]))
        np.save(res_dir + "/force_checkpoint.npy", np.array([lift_coeff_array, force_l_array, drag_coeff_array, force_d_array, time]))

    if t > plot_time_fig:     
        
        # Plot solution
        plt.figure()
        plot(u1, title="Velocity")
        plt.savefig(res_dir + "/u" + repr(t) + ".png", dpi=300)
        plt.close()
            
        plt.figure()
        plot(p1, title="Pressure")
        plt.savefig(res_dir + "/p" + repr(t) + ".png", dpi=300)
        plt.close()

        plot_time_fig += T/plot_freq_fig


    # Update time step and variables/functions using dt
    u0.assign(u1)
    dt.assign(0.99*project(dt_sf, DG0).vector().min())
    o0.assign(cos(omega))
    o1.assign(sin(omega))
    w0.assign(interpolate(w_bc, V))

    t += dt((5,7))

if(shut_down):
    sys.exit()
else:
    # Save force arrays
    np.save(res_dir + "/lift_force.npy", force_l_array)
    np.save(res_dir + "/lift_coefficient.npy", lift_coeff_array)
    np.save(res_dir + "/drag_force.npy", force_d_array)
    np.save(res_dir + "/drag_coefficient.npy", drag_coeff_array)
    np.save(res_dir + "/time.npy", time)

    # Plot solution
    plt.figure()
    plot(u1, title="Velocity, " + s + ", Re = " + repr(int(re)))
    plt.savefig(res_dir + "/u" + repr(int(t)) + '.png', dpi=300)

    plt.figure()
    plot(p1, title="Pressure, " + s + ", Re = " + repr(int(re)))
    plt.savefig(res_dir + "/p" + repr(int(t)) + '.png', dpi=300)

    # Plot the lift force
    plt.figure()
    plt.title("Lift Coefficient, " + s + ", Re = " + repr(int(re)))
    plt.plot(time, lift_coeff_array)

    # Plot the average of the lift force during the last 10 seconds
    lift_avg = np.array([sum(lift_coeff_array[int(len(lift_coeff_array) * (T-10)/T):]) / len(lift_coeff_array[int(len(lift_coeff_array) * (T-10)/T):])]*len(time))
    plt.plot(time, lift_avg, color='red', label='avg = ' + "{:.3f}".format(lift_avg[0])) 
    plt.legend()
    plt.savefig(res_dir + "/lift_force" + repr(int(t)) + '.png', dpi=300)

    # Plot the drag force
    plt.figure()
    plt.title("Drag Coefficient, " + s + ", Re = " + repr(re))
    plt.plot(time, drag_coeff_array)

    # Plot the average of force_array_2 during the last 5 seconds
    drag_avg = np.array([sum(drag_coeff_array[int(len(drag_coeff_array) * (T-5)/T):]) / len(drag_coeff_array[int(len(drag_coeff_array) * (T-5)/T):])]*len(time))
    plt.plot(time, drag_avg, color='red', label='avg = ' + "{:.3f}".format(drag_avg[0])) 
    plt.legend()
    plt.savefig(res_dir + "/drag_force" + repr(int(t)) + '.png', dpi=300)

    plt.close('all')

print("Adaptive Simulation Time:", TIME.time() - start_time, "Using args:", des, rpm, res)