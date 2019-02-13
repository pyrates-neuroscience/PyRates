import time as t
from pyrates.utility.cluster_compute import ClusterGridSearch

nodes = [
        'animals',
        'spanien',
        'carpenters',
        'osttimor'
        ]

compute_dir = "/nobackup/spanien1/salomon/ClusterGridSearch/CGS_Test"

t0 = t.time()

# cc = ClusterComputeTest(nodes, compute_dir)
cgs = ClusterGridSearch(nodes, compute_dir=compute_dir)

# Run 'ls' command on all nodes
cgs.run(command="ls")

print(f'Overall elapsed time: {t.time()-t0:.3f} seconds')
