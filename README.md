# DD2365-ME-Project
Project for the course DD2365 Advanced Computations In Fluid Dynamics. The project is using an ALE-FEM method to simulate the air-flow and estimate the Magnus effect on different designs of Flettner rotors.


Run FEniCS dolfin docker to run the code:
```
sudo docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:current
cd shared
```

Run either a single simulation or several in parallel using:

Single using design 0 and rpm = 40:
```
python3 ALE_Magnus.py 0 40
```

In parallel using design 0 & 1 and rpm = 40, 80 & 120:
```
parallel python3 ALE_Magnus.py ::: 0 1 ::: 40 80 120
```
