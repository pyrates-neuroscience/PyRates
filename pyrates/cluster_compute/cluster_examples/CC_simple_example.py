# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network models and simulations. See also:
# https://github.com/pyrates-neuroscience/PyRates
#
# Copyright (C) 2017-2018 the original authors (Richard Gast and
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain
# Sciences ("MPI CBS") and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# CITATION:
#
# Richard Gast and Daniel Rose et. al. in preparation

"""
Simple example how to use ClusterCompute for arbitrary purpose.

"""

from pyrates.cluster_compute.cluster_compute import *


class ClusterComputeExample(ClusterCompute):
    # Overwrite thread_master function
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
