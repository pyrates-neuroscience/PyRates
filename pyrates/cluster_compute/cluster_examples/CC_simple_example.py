"""
Simple example how to use ClusterCompute for arbitrary purpose.

"""

from pyrates.cluster_compute.cluster_compute import *


class ClusterComputeExample(ClusterCompute):
    def thread_master(self, client, thread_kwargs: dict):
        pm_client = client["paramiko_client"]
        logfile = client["logfile"]

        command = thread_kwargs["command"]

        # Execute 'command' on each remote worker without switching threads in between
        with self.lock:
            stdin, stdout, stderr = pm_client.exec_command(command +
                                                           f' &>> {logfile}',
                                                           get_pty=True)
        # Wait for remote execution to finish
        stdout.channel.recv_exit_status()


if __name__ == "__main__":
        nodes = [
                'animals',
                # 'spanien',
                'carpenters',
                'osttimor'
                ]

        compute_dir = "/nobackup/spanien1/salomon/ClusterCompute/CC_simple_example"

        cce = ClusterComputeExample(nodes, compute_dir=compute_dir)

        # Run 'ls' command on all nodes
        # All stdout (e.g. prints) will be written to each node logfile in the compute directory
        cce.run(thread_kwargs={
                    "command": "ls"
                })
