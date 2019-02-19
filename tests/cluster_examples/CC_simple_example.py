"""
Simple example how to use ClusterCompute for arbitrary purpose.

"""

from pyrates.utility.cluster_compute import *


class ClusterComputeExample(ClusterCompute):
    def run(self, **kwargs):
        t0 = t.time()

        # Insert arbitrary prepocessing here
        command = kwargs["command"]

        threads = [self.spawn_thread(client, command=command) for client in self.clients]
        for t_ in threads:
            t_.join()

        print("")
        print(f'Cluster computation finished. Elapsed time: {t.time()-t0:.3f} seconds')

    def thread_master(self, client, kwargs_: dict):
        thread_name = currentThread().getName()
        pm_client = client["paramiko_client"]
        logfile = client["logfile"]

        command = kwargs_["command"]

        # Execute 'command' on each remote worker
        with self.lock:
            stdin, stdout, stderr = pm_client.exec_command(command +
                                                           f' &>> {logfile}',
                                                           get_pty=True)
        # Wait for remote execution to finish
        stdout.channel.recv_exit_status()


if __name__ == "__main__":
        nodes = [
                'animals',
                'spanien',
                'carpenters',
                'osttimor'
                ]

        compute_dir = "/nobackup/spanien1/salomon/ClusterCompute/CC_simple_example"

        cce = ClusterComputeExample(nodes, compute_dir=compute_dir)

        # Run 'ls' command on all nodes
        # All stdout (e.g. prints) will be written to each nodes logfile in the compute directory
        cce.run(command="ls")
