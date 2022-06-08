# DD2365-ME-Project
Project for the course DD2365 Advanced Computations In Fluid Dynamics. The project is using an ALE-FEM method to simulate the airflow and estimate the Magnus effect on different designs of Flettner rotors. The latest python script that should be used is ALE_adaptive.py.


Run FEniCS Dolfin docker to run the code:
```
sudo docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:current
cd shared
```

Run either a single simulation or several in parallel using:

Single using design 0 and rpm = 40 with resolution = 32:
```
python3 ALE_adaptive.py 0 40 32
```

In parallel using design 0 & 1 and rpm = 40, 80 & 120 with resolution 32:
```
parallel python3 ALE_adaptive.py ::: 0 1 ::: 40 80 120 ::: 32
```

