import numpy as np
from matplotlib import pyplot as plt

l_c = "lift_coefficient.npy" #"lift_force.npy"
t_s = "time.npy"
res_dir = "final_plots"

exp_lift_coeff = [2.4, 5.5, 8.5, 10.75, 12.2]

l = []
t = []

l_r = []
t_r = []

#l.append(np.load("16_results_ALE_0_48_2805049.0883590463/"+l_c))
#t.append(np.load("16_results_ALE_0_48_2805049.0883590463/"+t_s))

#l.append(np.load("32_results_ALE_0_48_2805049.0883590463/"+l_c))
#t.append(np.load("32_results_ALE_0_48_2805049.0883590463/"+t_s))

#l.append(np.load("48_results_ALE_0_48_2805049.0883590463/"+l_c))
#t.append(np.load("48_results_ALE_0_48_2805049.0883590463/"+t_s))

#l.append(np.load("64_results_ALE_0_48_2805049.0883590463/"+l_c))
#t.append(np.load("64_results_ALE_0_48_2805049.0883590463/"+t_s))

for i in range(16, 64+1, 8):
    s = "0_final_results_0_239_" + str(i) +"_2805049/"
    l.append(np.load(s+l_c))
    t.append(np.load(s+t_s))

for j in [0, 1, 2]:
    l_r.append([])
    t_r.append([])
    for i in [48, 96, 143, 191, 239]:
        s = "0_final_results_" + str(j) + "_" + str(i) + "_60_2805049/"
        
        if j == 0 and i == 239: s = "0_final_results_" + str(j) + "_" + str(i) + "_64_2805049/"
        if i > 144: 
            l_r[j].append(np.load(s+"force_checkpoint.npy")[0])
            t_r[j].append(np.load(s+"force_checkpoint.npy")[4])
        else:
            l_r[j].append(np.load(s+l_c))
            t_r[j].append(np.load(s+t_s))

lift_2 = []
lift_2_val = []
for j in range(len(l_r)):
    lift_2.append([])
    lift_2_val.append([])
    for i in range(len(l_r[j])):
        # Plot the lift force
        #plt.figure()
        #plt.title("Lift Coefficient, " + str(16 + 8*i))
        #plt.plot(t[i], l[i], linewidth=0.2)

        a = 10
        lift_2[j].append(np.array([sum(l_r[j][i][int(len(l_r[j][i]) * (t_r[j][i][-1]-a)/t_r[j][i][-1]):]) / len(l_r[j][i][int(len(l_r[j][i]) * (t_r[j][i][-1]-a)/t_r[j][i][-1]):])]*len(t_r[j][i])))
        lift_2_val[j].append(lift_2[j][i][0])
        #plt.plot(t[i], lift_2[i], color='red', label='avg = ' + "{:.3f}".format(lift_avg[i][0])) 
        #plt.legend()
        #plt.savefig(res_dir + "/lift_coeff" + str(16 + 8*i) + '.png', dpi=300)

plt.figure()
for j in range(len(l_r)):
    c = "blue"
    if j == 0: 
        c = "blue"
        m = "o"
        la = "circle"
    elif j == 1: 
        c = "orange"
        m = "+"
        la = "square"
    elif j == 2: 
        c = "green"
        m = "x"
        la = "circle +"
    #plt.title("Lift Coefficient, Resolution")
    s_r = np.array([1, 2, 3, 4, 5])
    #print(s_r, lift_2_val)
    plt.plot(s_r, lift_2_val[j], linewidth=1, label=la, color=c)
    plt.plot(s_r, lift_2_val[j], m, linewidth=1, color=c) # Add markers
    plt.legend()

plt.xlabel("Spin Ratio")
plt.ylabel("Lift Coefficient")
plt.savefig(res_dir + "/lift_coeff_rpm" + '.png', dpi=300)

plt.figure()
plt.plot(s_r, lift_2_val[0], linewidth=1, label="CFD", color="blue")
plt.plot(s_r, lift_2_val[0], "o", linewidth=1, color="blue") # Add markers
plt.legend()
plt.plot(s_r, exp_lift_coeff, linewidth=1, label="Experimental data", color="orange")
plt.plot(s_r, exp_lift_coeff, "+", linewidth=1, color="orange") # Add markers
plt.legend()

plt.xlabel("Spin Ratio")
plt.ylabel("Lift Coefficient")
plt.savefig(res_dir + "/lift_coeff_exp_rpm" + '.png', dpi=300)


lift_avg = []
lift_avg_val = []
for i in range(len(l)):
    # Plot the lift force
    plt.figure()
    plt.title("Lift Coefficient, Resolution = " + str(16 + 8*i))
    plt.plot(t[i], l[i], linewidth=0.7)
    plt.xlabel("Time, seconds")
    plt.ylabel("Lift Coefficient")

    # Plot the average of the lift force_array
    #lift_avg = np.array([sum(lift_coeff_array) / len(lift_coeff_array)]*len(time))
    #plt.plot(time, lift_avg, color='red', label='avg = ' + "{:.3f}".format(lift_avg[0]))
    #plt.legend()
    #plt.savefig(res_dir + "/lift_force" + repr(int(t)) + '.png', dpi=300)
    # Plot the average of the lift force during the last a seconds
    a = 7
    lift_avg.append(np.array([sum(l[i][int(len(l[i]) * (t[i][-1]-a)/t[i][-1]):]) / len(l[i][int(len(l[i]) * (t[i][-1]-a)/t[i][-1]):])]*len(t[i])))
    lift_avg_val.append(lift_avg[i][0])
    plt.plot(t[i], lift_avg[i], color='red', label='avg = ' + "{:.3f}".format(lift_avg[i][0])) 
    plt.legend()
    plt.savefig(res_dir + "/lift_coeff" + str(16 + 8*i) + '.png', dpi=300)

plt.figure()
#plt.title("Lift Coefficient, Resolution")
res = np.array([16+i*8 for i in range(0,7)])
print(res, lift_avg_val)
plt.plot(res, lift_avg_val, linewidth=1, color="blue")
plt.plot(res, lift_avg_val, "+", linewidth=1, color="blue") # Add markers
plt.xlabel("Resolution")
plt.ylabel("Lift Coefficient")
plt.savefig(res_dir + "/lift_coeff_res" + '.png', dpi=300)

"""
theo_lift = []
spin_ratio = []
for w in range(0, 239):
    Vr = 2 * np.pi * 6.561680 * (w / 60)
    G = 2.0 * np.pi * 6.561680 * Vr
    L = 0.002418803318304776 * 32.808398950131235 * G # lbs/ft
    theo_lift.append(L * 1.3558179483) # From lbs/ft to nm
    spin_ratio.append(2.0 * (w * 2 * np.pi / 60) / 10.0)


plt.figure()
plt.title("Theoretical Lift Force")
plt.plot(spin_ratio, theo_lift, linewidth=1)
plt.savefig(res_dir + "/theoretical_lift_coeff" + '.png', dpi=300)
"""