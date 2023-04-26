import numpy as np
import matplotlib.pyplot as plt


# Define the birth-death process
lam = 1.5
mu = 2

# Define the initial probability distribution
P0 = np.array([0.0, 1.0, 0.0,0.0])

# Define the time interval and step size
t0 = 0.0
tf = 15.0
dt = 1/60
Nt = int((tf-t0)/dt)

# Define the discretized Kolmogorov forward equation
def kolmogorov_forward(P, t):
    dP = np.zeros_like(P)
    dP[0] = -lam*P[0] + mu*P[1]
    dP[1] = mu*P[2] + lam*P[0] - lam*P[1] - mu*P[1] 
    dP[2] = mu*P[3] + lam*P[1] - lam*P[2] - mu*P[2] 
    dP[3] = lam*P[2] - mu*P[3]
    return dP

# Initialize the solution
P = np.zeros((Nt+1, 4))
P[0] = P0

# Apply the fourth-order Runge-Kutta method
for i in range(Nt):
    k1 = dt*kolmogorov_forward(P[i], t0 + i*dt)
    k2 = dt*kolmogorov_forward(P[i] + 0.5*k1, t0 + i*dt + 0.5*dt)
    k3 = dt*kolmogorov_forward(P[i] + 0.5*k2, t0 + i*dt + 0.5*dt)
    k4 = dt*kolmogorov_forward(P[i] + k3, t0 + i*dt + dt)
    P[i+1] = P[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

# Plot the solution
result = np.zeros((Nt+1, 4))
result = (1 - P[:0])
total = 0
for i in range(Nt):
    total+=(1-P[i,3])*mu

print("gi(s,0,15) : {}".format(total/(mu*Nt)))


plt.plot(np.linspace(t0, tf, Nt+1), P[:,0], label='P0')
plt.plot(np.linspace(t0, tf, Nt+1), P[:,1], label='P1')
plt.plot(np.linspace(t0, tf, Nt+1), P[:,2], label='P2')
plt.plot(np.linspace(t0, tf, Nt+1), P[:,3], label='P3')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.show()
