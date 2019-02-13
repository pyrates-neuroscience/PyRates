
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
"""Functions for performing computations in a compute cluster with pyrates models.
"""
# TODO: Why does it take so long to start the master/worker script? PyRates?
# system imports
import os
import sys
import json
import time as t
import glob
import getpass
from shutil import copy2
from pathlib import Path
from datetime import datetime
from threading import Thread, currentThread, RLock

# external imports
import pandas as pd
import numpy as np
import paramiko

# pyrates internal imports
# from pyrates.utility.grid_search import linearize_grid


# meta infos
__author__ = "Christoph Salomon"
__status__ = "development"


class StreamTee(object):
    # TODO: Stop stream tee after CGS computation has finished

    """Copy all stdout to a specified file"""
    def __init__(self, stream1, stream2fp):
        stream2 = open(stream2fp, "a")
        self.stream1 = stream1
        self.stream2 = stream2
        self.__missing_method_name = None  # Hack!

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name  # Could also be a property
        return getattr(self, '__methodmissing__')

    def __methodmissing__(self, *args, **kwargs):
        # Emit method call to the log copy
        callable2 = getattr(self.stream2, self.__missing_method_name)
        callable2(*args, **kwargs)

        # Emit method call to stdout (stream 1)
        callable1 = getattr(self.stream1, self.__missing_method_name)
        return callable1(*args, **kwargs)


class ClusterCompute(object):
    def __init__(self, nodes, compute_dir=None):
        """Create new ClusterCompute instance object with unique compute ID

        Creates a compute directory for the ClusterCompute instance, either in the specified path or as a default
        folder in the current working directory of the executing script.
        Creates a logfile in ComputeDirectory/Logs and tees all future stdout and stderr to this file
        Connects to all nodes via SSH and saves the corresponding paramiko client that can be used to execute commands
        on the remote machines

        Parameters
        ----------
        nodes:
            List of names or IP addresses of working stations/servers in the local network
        compute_dir:
            Full path to a directory that will be used as compute directory.
            If none is given, a default compute directory is created in the current working directory

        Returns
        -------
        ClusterCompute instance object

        """

        self.clients = []
        self.lock = RLock()

        #############################################################
        # Create main compute directory, subdirectories and logfile #
        #############################################################

        # Unique compute ID
        self.compute_id = datetime.now().strftime("%d%m%y-%H%M%S")

        # Main compute directory
        if compute_dir:
            self.compute_dir = compute_dir
            os.makedirs(self.compute_dir, exist_ok=True)
        else:
            # Create default compute directory in the current working directory
            self.compute_dir = f'{os.getcwd()}/ClusterCompute_{self.compute_id}'
            os.makedirs(self.compute_dir, exist_ok=True)

        # Logfile directory
        self.log_dir = f'{self.compute_dir}/Logs/{self.compute_id}'
        os.makedirs(self.log_dir, exist_ok=True)

        # Global logfile to copy stdout and stderr to
        self.global_logfile = f'{self.log_dir}/Global_logfile.log'
        os.makedirs(os.path.dirname(self.global_logfile), exist_ok=True)

        ####################################
        # Tee stdout and stderr to logfile #
        ####################################
        sys.stdout = StreamTee(sys.stdout, self.global_logfile)
        sys.stderr = StreamTee(sys.stderr, self.global_logfile)

        ###########################
        # Write header to logfile #
        ###########################
        print("***NEW CLUSTER INSTANCE CREATED***")
        print(f'Compute ID: {self.compute_id}')
        print(f'Compute directory: {self.compute_dir}')
        print(f'Global logfile: {self.global_logfile}')
        print("")

        ##################
        # Create cluster #
        ##################
        print("***CONNECTING TO NODES...***")
        t0 = t.time()
        self.cluster_connect(nodes)
        print(f'Nodes connected. Elapsed time: {t.time()-t0:.3f} seconds')

    def run(self, **kwargs):
        """Start a thread for each connected client. Each thread executes the __thread_master() function

        Each thread and therefor each instance of the __thread_master() function is responsible for the communication
        with one node in the cluster. All kwargs arguments are passed as a dict to the __thread_master() function via
        kwargs_. Additional kwargs can be passed to the thread_master via **kwargs.
        Stops execution of the outside script until all threads have finished.
        Can be called multiple times from the same ClusterCompute instance

        !!! Can be overwritten in by inherited classed to fit the needs of the user !!!

        Params
        ------
        kwargs

        Returns
        -------

        """

        t0 = t.time()

        threads = [self.spawn_thread(client) for client in self.clients]
        for t_ in threads:
            t_.join()

        print("")
        print(f'Cluster computation finished. Elapsed time: {t.time()-t0:.3f} seconds')

    def spawn_thread(self, client, **kwargs):
        t_ = Thread(
            name=client["node_name"],
            target=self.thread_master,
            args=(client, kwargs)
        )
        t_.start()
        return t_

    def thread_master(self, client, kwargs_: dict):
        """Function that is executed by every thread. Every instance of thread_master is bound to a different client

        The __thread_master() function can be arbitrarily changed to fit the needs of the user.
        Commands on the node can be executed using client["paramiko_client"].exec_command()
        self.lock can be used to ensure a piece of code is executed without threads being switched
        e.g.
            with self.lock:
                some code that is executed without switching to another thread
        Since __thread_master() is called by __spawn_threads(), which again is called by run(), **kwargs of
        spawn_threads() have to be parsed as dict to thread_master(). kwargs of run() are NOT automatically passed to
        spawn_threads() or thread_master()

        Params
        ------
        client
            dict containing the paramiko client, the name of the connected node, some hardware specifications and the
            logfile
        kwargs_
            dict containing **kwargs of spawn_threads() as key/value pairs

        Returns
        -------

        """
        pass

    def cluster_connect(self, nodes):
        """Connect to all nodes via SSH

        Connect to all nodes in the given list via SSH, using __ssh_connect()
        Adds a dictionary for each node to the class internal 'clients' list
        Each dictionary contains:
            - ["paramiko_client"]: A paramiko client that can be used to execute commands on the node
            - ["node_name"]: The name of the connected node
            - ["hardware"]: A dictionary with certain hardware information of the node
            - ["logfile"]: A local logfile where all stdout and stderr of the node will be redirected to

        Parameters
        ----------
        nodes
            List with names or IP addresses of working stations/servers in the local network

        Returns
        -------

        """
        username = getpass.getuser()
        # password = getpass.getpass(prompt='Enter password: ', stream=sys.stderr)

        for node in nodes:
            client = self.ssh_connect(node, username=username)
            if client is not None:
                local_logfile = f'{self.log_dir}/Local_logfile_{node}_.log'
                os.makedirs(os.path.dirname(local_logfile), exist_ok=True)

                hw = self.get_hardware_spec(client)

                self.clients.append({
                    "paramiko_client": client,
                    "node_name": node,
                    "hardware": hw,
                    "logfile": local_logfile})
            print("")

    @staticmethod
    def get_hardware_spec(pm_client):
        stdin, stdout, stderr = pm_client.exec_command("lscpu | grep 'Model name'")
        cpu = stdout.readline().split()
        cpu = ' '.join(map(str, cpu[2:]))
        print(f'CPU: {cpu}')

        stdin, stdout, stderr = pm_client.exec_command("lscpu | grep 'CPU(s)'")
        num_cpu_cores = int(stdout.readline().split(":")[1])
        print(f'Cores: {num_cpu_cores}')

        stdin, stdout, stderr = pm_client.exec_command("lscpu | grep 'min MHz'")
        cpu_min = float(stdout.readline().split(":")[1])
        print(f'CPU min: {cpu_min} MHz')

        stdin, stdout, stderr = pm_client.exec_command("lscpu | grep 'max MHz'")
        cpu_max = float(stdout.readline().split(":")[1])
        print(f'CPU max: {cpu_max} MHz')

        stdin, stdout, stderr = pm_client.exec_command("free -m | grep 'Mem'")
        total_mem = int(stdout.readline().split()[1])
        print(f'Total memory: {total_mem} MByte')

        hw = {
            'cpu': cpu,
            'num_cpu_cores': num_cpu_cores,
            'cpu_min': cpu_min,
            'cpu_max': cpu_max,
            'total_mem': total_mem
        }
        return hw

    @staticmethod
    def ssh_connect(node, username, password=None):
        """Connect to a host via SSH

        Parameters
        ----------
        node
            Name or IP-address of the host to connect to
        username
        password
            ssh_connect uses kerberos authentication so no password is required. If kerberos is not available on your
            network, Paramiko needs a password to create the SSH connection
        Returns
        -------
        paramiko.SSHClient()
            Is None if connection fails. For detailed information of the thrown exception see Paramiko documentation

        """
        client = paramiko.SSHClient()
        # client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # Using Kerberos authentication to connect to remote machines
            # If Kerberos is not available in you cluster you need to implement an (unechoed) password request
            # (e.g. using getpass)
            client.connect(node, username=username, gss_auth=True, gss_kex=True)
            print(f'\'{node}\': Connection established!')
            return client
            # return client.invoke_shell()
        except paramiko.ssh_exception.NoValidConnectionsError as err:
            print(f'\'{node}\': ', err)
            return None
        except paramiko.ssh_exception.AuthenticationException as err:
            print(f'\'{node}\': ', err)
            return None
        except paramiko.ssh_exception.SSHException as err:
            print(f'\'{node}\': ', err)
            return None
        except IOError as err:
            print(f'\'{node}\': ', err)
            return None


class ClusterComputeTest(ClusterCompute):
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

        # Execute 'command' on the remote worker
        with self.lock:
            stdin, stdout, stderr = pm_client.exec_command(command +
                                                           f' &>> {logfile}',
                                                           get_pty=True)
        # Wait for remote execution to finish
        stdout.channel.recv_exit_status()


class ClusterGridSearch(ClusterCompute):
    def __init__(self, nodes, compute_dir=None):
        super().__init__(nodes, compute_dir)

        # Add additional subfolders to the compute directory

        # Grid directory
        self.grid_dir = f'{self.compute_dir}/Grids'
        os.makedirs(self.grid_dir, exist_ok=True)

        # Config directory
        self.config_dir = f'{self.compute_dir}/Config'
        os.makedirs(self.config_dir, exist_ok=True)

        # Result directory
        self.res_dir = f'{self.compute_dir}/Results'
        os.makedirs(self.res_dir, exist_ok=True)

    def run(self, **kwargs):
        # TODO: Implement ClusterGridSearch run functionality
        t0 = t.time()

        # Insert arbitrary prepocessing here
        command = kwargs["command"]

        threads = [self.spawn_thread(client,
                                     command=command) for client in self.clients]
        for t_ in threads:
            t_.join()

        print("")
        print(f'Cluster computation finished. Elapsed time: {t.time() - t0:.3f} seconds')

    def thread_master(self, client, kwargs_: dict):
        # TODO: Implement ClusterGridSearch thread_master
        thread_name = currentThread().getName()
        pm_client = client["paramiko_client"]
        logfile = client["logfile"]

        command = kwargs_["command"]

        # Execute 'command' on the remote worker
        with self.lock:
            stdin, stdout, stderr = pm_client.exec_command(command +
                                                           f' &>> {logfile}',
                                                           get_pty=True)
        # Wait for remote execution to finish
        stdout.channel.recv_exit_status()


def linearize_grid(grid: dict, permute=False):
    """
    Parameters
    ----------
    grid
    permute
    Returns
    -------
    """

    arg_lengths = [len(arg) for arg in grid.values()]

    if len(list(set(arg_lengths))) == 1 and not permute:
        return pd.DataFrame(grid)
    else:
        vals, keys = [], []
        for key, val in grid.items():
            vals.append(val)
            keys.append(key)
        new_grid = np.stack(np.meshgrid(*tuple(vals)), -1).reshape(-1, len(grid))
        return pd.DataFrame(new_grid, columns=keys)
