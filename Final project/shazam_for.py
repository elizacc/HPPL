from mpi4py import MPI
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


def get_signal(a,b,c,d,e,f,g,h,j,k):
    t=np.linspace(-20*2*np.pi, 20*2*np.pi, 2**10)
    y=np.sin(a*t)*np.exp(-(t-b*2*np.pi)**c/d/e**2)
    y=y+np.sin(f*t)*np.exp(-(t-g*2*np.pi)**h/j/k**2)
    return y

np.random.seed(8)
signals = []
for i in range(100):
    a, f = np.random.randint(1,6,size=2)
    b, g = np.random.randint(-10, 10, size=2)
    c, h = np.random.randint(1, 3, size=2)
    d, j = np.random.randint(1, 4, size=2)
    e, k = 10 * np.random.randint(1, 4, size=2)
    signals.append(get_signal(a,b,c,d,e,f,g,h,j,k))
    
new_signal = get_signal(3, 0, 2, 2, 20, 1, 5, 2, 2, 20)


#MPI
start =  MPI.Wtime()

comm = MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

nwindowsteps=1000
window_steps = []
for n in range(rank*(nwindowsteps//size), (rank+1)*(nwindowsteps//size)):
    window_steps.append(t[0] + (t[-1] - t[0]) * n / nwindowsteps)

ys = signals.copy()
ys.append(new_signal)

if root == 0:
    specgrams = []
for y in ys:
    specgram = []
    for i, window_position in enumerate(window_steps):
        y_window = []
        for j in range(len(t)):
            y_window.append(np.exp(-(t[j] - window_position)**2/2/window_width**2) * y[j])
        specgram.append(abs(np.fft.fft(y_window)))
    final_specgram = np.array(comm.gather(specgram, root=0))
    if root==0:
        specgrams.append(final_specgram)

dists = []
for i in range(rank*(len(signals)//size), (rank+1)*(len(signals)//size)):
    dist = 0
    for n in range(nwindowsteps):
        for j in range(len(t)):
            dist += (specgrams[-1][n][j] - specgrams[i][n][j])**2
    dists.append(dist)

final_dists = np.array(comm.gather(dists, root=0))
if root==0:
    best_match = np.argsort(final_dists)[0]

end = MPI.Wtime()

if rank==0:
    print(best_match)
    print(end-start)
