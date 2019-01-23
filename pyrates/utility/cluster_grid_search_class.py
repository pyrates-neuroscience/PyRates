
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
"""Functions for performing parameter grid simulations in a compute cluster with pyrates models.
"""

# external imports
import pandas as pd
import numpy as np
import paramiko

# system imports
import os
import sys
import json
import tkinter as tk
import tkinter.simpledialog

import getpass
from pathlib import Path
from datetime import datetime
from threading import Thread, currentThread, RLock

# pyrates internal imports
from pyrates.utility.grid_search import linearize_grid

# meta infos
__author__ = "Christoph Salomon"
__status__ = "development"


# Class to copy stdout and stderr to a given file
class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class ClusterGridSearch(object):
    def __init__(self, nodes, global_config, **kwargs):
        self.nodes = nodes
        self.global_config = global_config
        self.clients = []
        self.param_grid = None

        # Create compute directory, log directory and global logfile
        ############################################################

        # Unique id of current ClusterGridSearch instance.
        self.compute_id = datetime.now().strftime("%d%m%y-%H%M%S")

        # Name of computation as a combination of global config filename and unique compute id
        self.compute_name = f'{Path(self.global_config).stem}_{self.compute_id}'

        # Create compute directory as subdir of global_config location
        self.compute_dir = f'{os.path.dirname(self.global_config)}/{self.compute_name}'
        os.makedirs(os.path.dirname(self.compute_dir), exist_ok=True)

        # Create log directory as subdir of compute directory
        self.log_dir = f'{self.compute_dir}/Logs'
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)

        # Create result directory as subdir of compute directory
        self.res_dir = f'{self.compute_dir}/Results'
        os.makedirs(os.path.dirname(self.res_dir), exist_ok=True)

        # Create global logfile in log dir
        self.global_logfile = f'{self.log_dir}/Global_log_{self.compute_name}.log'
        os.makedirs(os.path.dirname(self.global_logfile), exist_ok=True)

        # Copy all future stdout to logfile
        sys.stdout = Logger(self.global_logfile)

        print(f'New ClusterGridSearch instance created!')
        print(f'Compute ID: {self.compute_id}')
        print(f'Compute directory: {self.compute_dir}')
        print(f'Log directory: {self.log_dir}')
        print(f'Result directory: {self.res_dir}')
        print(f'Global logfile: {self.global_logfile}')

        # Create compute cluster
        ########################

        print("Creating cluster...")

        # Get password to connect to cluster nodes.
        username = getpass.getuser()
        password = getpass.getpass(prompt='Enter password:\n', stream=sys.stderr)

        for node in self.nodes['hostnames']:
            client = (self.__ssh_connect(node, username=username, password=password))
            #   client[0] contains the client handler
            #   client[1] contains the corresponding nodename
            if client is not None:
                self.clients.append(client)

        # TODO: Print/save hardware specifications of each node

        # Create lock to control thread scheduling
        self.lock = RLock()

    def __del__(self):
        for client in self.clients:
            client[0].close()

    def compute_grid(self, param_grid, local_config=None):
        # Create parameter grid from either .csv, dict or DataFrame
        if isinstance(param_grid, str):
            print(f'Loading parameter grid: {param_grid}... ', end="")
            try:
                self.param_grid = pd.read_csv(param_grid)
                print("done!")
            except FileNotFoundError as err:
                print(err)
                return
        elif isinstance(param_grid, dict):
            self.param_grid = linearize_grid(param_grid, permute=True)
        else:
            self.param_grid = param_grid

        # TODO: Check if param_grid and param_map match

        # Add status flag
        self.param_grid['status'] = 'unsolved'
        print("Created parameter grid:")
        print(f'{self.param_grid}')

        # Start threadpool
        ##################

        # TODO: Implement asynchronous computation instead of multithreading

        # TODO: Something is fishy when calling __spawn_thread() with self instead of classname
        # error message concerning inputs
        threads = [self.__spawn_thread(self, client, local_config) for client in self.clients]
        # works but I dont like it
        # threads = [ClusterGridSearch.__spawn_thread(self, client) for client in self.clients]

        # Wait for all threads to finish
        for t in threads:
            t.join()

    def __scheduler(self, client, local_config):

        thread_name = currentThread().getName()

        # dummy
        stdin, stdout, stderr = client.exec_command('ls', get_pty=True)

        for line in iter(stdout.readline, ""):
            print(f'[H]\'{thread_name}\': {line}', end="")


        # Check if create_ssh_connection() returned 0
        # if client:
        #     # Check if 'status'-key is present in param_grid
        #     if not fetch_param_idx(param_grid, set_status=False).isnull():
        #
        #         # host_cmd['host_env']: Path to python executable inside a conda environment with installed packages:
        #         #   'pandas', 'pyrates'
        #         # host_cmd['host_file']: Path to python script to execute on the remote host
        #         command = host_cmd['host_env'] + ' ' + host_cmd['host_file']
        #
        #         # TODO: Automatically create a folder where the result file is located to put node output files to
        #         result_path = f'/data/hu_salomon/Documents/ClusterGridSearch/Results/test/'
        #
        #         # TODO: Copy environment, script and config to shared or local directory on the remote host
        #         # Change paths of host_env and host_file respectively
        #
        #         # TODO: Call exec_command only once and send global config file
        #         # TODO: Send node specific config to stdin inside a loop
        #         # stdin.write()
        #         # stdin.flush()
        #
        #         # Scheduler:
        #         # fetch_param_idx() returns empty index if all parameter combinations have been calculated
        #         # lock.acquire()
        #         while not fetch_param_idx(param_grid, lock, set_status=False).empty:
        #
        #             # Make sure all of the following commands are executed before switching to another thread
        #             # lock.acquire()
        #             with lock:
        #                 print(f'[T]\'{thread_name}\': Fetching index... ', end="")
        #
        #                 # Get index of a parameter combination that hasn't been computed yet
        #                 param_idx = fetch_param_idx(param_grid, num_params=2)
        #
        #                 # Get parameter combination to pass as argument to the remote host
        #                 param_grid_arg = param_grid.iloc[param_idx]
        #
        #                 print(*param_idx)
        #                 # print(f'{param_idx}')
        #                 print(f'[T]\'{thread_name}\': Starting remote computation')
        #
        #                 # TODO: Create node specific config file for each computation and send it to the node
        #                 stdin, stdout, stderr = client.exec_command(command +
        #                                                             f' --global_config={config_file}'
        #                                                             f' --local_config=""'
        #                                                             f' --param_grid_arg="{param_grid_arg.to_dict()}"'
        #                                                             f' --result_path={result_path}',
        #                                                             get_pty=True)
        #
        #             # While waiting for the remote computation to finish, other threads can now be active
        #             # lock.release()
        #
        #             # Wait for remote computation to finish
        #             # exit_status = stdout.channel.recv_exit_status()
        #
        #             # node_result.from_csv(node_result_file)
        #
        #             # Print what has been sent to the channels stdout or stderr
        #             for line in iter(stdout.readline, ""):
        #                 print(f'[H]\'{thread_name}\': {line}', end="")
        #
        #             # TODO: Check if resultfile has been created and is not empty. If so, change status of current
        #             #  param_idx in param_grid from 'pending' to 'done'. Otherwise set status to 'failed'
        #
        #     else:
        #         # If no key named 'status' in param_grid:
        #         print(f'[T]\'{host}\': "No key named \'status\' in param_grid')
        #
        #     # TODO: If no shared memory is available, copy result files from host back to local workstation
        #
        #     # TODO: Change current param_idx in param_grid from 'pending' to 'done'
        #
        #     client.close()
        # else:
        #     # If no connection to node was established:
        #     return

    def __spawn_thread(self, client):
        # client[0]: Paramiko client, client[1]: name of corresponding node
        t = Thread(
            name=client[1],
            # target=self.__scheduler,
            target=ClusterGridSearch.__scheduler,
            args=(self, client[0])
        )
        t.start()
        return t

    @staticmethod
    def __ssh_connect(node, username, password):
        """Connect to a host via SSH

        Parameters
        ----------
        nodename
            Name or IP-address of the host to connect to
        username
        password

        Returns
        -------
        paramiko.SSHClient()
            Throws exception and returns 0 if connection fails. See Paramiko documentation


        """
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # print(f'Connecting to host \'{host}\'...')
        try:
            client.connect(node, username=username, password=password)
            print(f'\'{node}\': Connection established')
            return client, node
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

    @staticmethod
    def __fetch_param_idx(param_grid, lock=None, num_params=1, set_status=True):
        """Fetch a pandas.Index([index_list]) with the indices of the first num_params rows of param_grid who's
        'status'-key equals 'unsolved'

        Parameters
        ----------
        param_grid
            Linearized parameter grid of type pandas.DataFrame.
        num_params
            Number of indices to fetch from param_grid. Is 1 by default.
        set_status
            If True, sets 'status' key of the fetched rows to 'pending', to exclude them from future calls.
            Can be used to check param_grid for fetchable or existend keys without changing their 'status' key.
            Is True by default.

        Returns
        -------
        pandas.Index([index_list])
            Is empty if there are no row indices to be fetched.
            Is np.nan if param_grid has no key named 'status'.
            Contains all remaining indices if num_params is higher than fetchable row indices.


        """
        if lock:
            with lock:
                try:
                    # Get the first num_params row indices of lin_grid who's 'status' keys equal 'unsolved'
                    param_idx = param_grid.loc[param_grid['status'] == 'unsolved'].index[:num_params]
                except KeyError:
                    return pd.Index([np.nan])
                if set_status:
                    param_grid.at[param_idx, 'status'] = 'pending'
                return param_idx
        else:
            try:
                # Get the first num_params row indices of lin_grid who's 'status' keys equal 'unsolved'
                param_idx = param_grid.loc[param_grid['status'] == 'unsolved'].index[:num_params]
            except KeyError:
                return pd.Index([np.nan])
            if set_status:
                param_grid.at[param_idx, 'status'] = 'pending'
            return param_idx

    def __set_param_result_status(self):
        pass


def create_cgs_config(filepath, circuit_template, param_map, dt, simulation_time, inputs,
                      outputs, sampling_step_size=None, permute_grid=False, **kwargs):
    """Create a configfile.json containing a config_dict{} with input parameters as key-value pairs

    Parameters
    ----------
    filepath
    circuit_template
    param_map
    dt
    simulation_time
    inputs
    outputs
    sampling_step_size
    permute_grid
    kwargs

    Returns
    -------

    """

    # TODO: Eliminate redundant brackets in inputs in config_file
    config_dict = {
        "circuit_template": circuit_template,
        "param_map": param_map,
        "dt": dt,
        "simulation_time": simulation_time,
        "inputs": {str(*inputs.keys()): list(*inputs.values())},
        "outputs": outputs,
        "sampling_step_size": sampling_step_size,
        "permute_grid": permute_grid,
        "kwargs": kwargs
    }

    with open(filepath, "w") as f:
        json.dump(config_dict, f, indent=2)
