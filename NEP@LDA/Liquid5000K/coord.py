import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList
from collections import defaultdict

# Parameters
xyz_file = 'traj.xyz'
cutoff = 1.85/2 # Carbon-carbon bond threshold

# Read all frames
atoms_list = read(xyz_file, index=':')

# Initialize counters
coord_counts_total = defaultdict(int)
n_frames = len(atoms_list)
print("Number of Frame:", n_frames)

for atoms in atoms_list:
    # Initialize neighbor list for current frame
    cutoffs = [cutoff] * len(atoms)
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)

    # Count coordination for each atom
    frame_counts = defaultdict(int)
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        # for j in indices:
        #    print(j, atoms.get_distance(i, j, mic=True)) 
        # exit()
        coordination = len(indices)
        # print("Atom", i, "neigh", coordination)
        if coordination in [2, 3, 4]:
            frame_counts[coordination] += 1

    # Update total counts
    for k in frame_counts:
        coord_counts_total[k] += frame_counts[k]

# Average over all frames
print("Average coordination numbers per frame:")
for k in [2, 3, 4]:
    avg = coord_counts_total[k] *100. / n_frames /len(atoms)
    print(f"{k}-fold coordinated: {avg:.2f} %")
