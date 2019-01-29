
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

# system imports
import os
import sys
import json
import time
import getpass
from pathlib import Path
from datetime import datetime
from threading import Thread, currentThread, RLock

# external imports

import pandas as pd
import numpy as np
import paramiko

# pyrates internal imports
from pyrates.utility.grid_search import linearize_grid


# meta infos
__author__ = "Christoph Salomon"
__status__ = "development"


# Class to copy stdout and stderr to a given file
class Logger(object):
    # TODO: Stdout is only written to logfile after computation has finished. Need to write output as soon as it appears
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class ClusterGridSearch(object):
    def __init__(self, global_config, compute_dir=None, **kwargs):
        print("Starting cluster grid search!")

        self.global_config = global_config

        # self members changed outside the __init__
        self.nodes = []
        self.clients = []
        self.lock = None
        self.param_grid_count = 0

        #############################################################
        # Create main compute directory, subdirectories and logfile #
        #############################################################
        print("")
        print("***CREATE WORKING DIRECTORY***")
        start_dir = time.time()

        # Unique id of current ClusterGridSearch instance.
        self.compute_id = datetime.now().strftime("%d%m%y-%H%M%S")

        # Name of computation as a combination of global config filename and unique compute id
        # Used for unique config file and default compute directory if none is given
        self.compute_name = f'{Path(self.global_config).stem}_{self.compute_id}'

        if compute_dir:
            # Use given dir as main directory
            self.compute_dir = compute_dir
            self.compute_id = datetime.now().strftime("%d%m%y-%H%M%S")
            print(f'Using existent directory: {self.compute_dir}')
        else:
            # Create default compute directory as subdir of global_config location
            self.compute_dir = f'{os.path.dirname(self.global_config)}/{self.compute_name}'
            os.makedirs(self.compute_dir, exist_ok=True)
            print(f'New working directory created: {self.compute_dir}')

        # Create log directory '/Logs' as subdir of the compute directory if it doesn't exist
        self.log_dir = f'{self.compute_dir}/Logs/{self.compute_id}'
        os.makedirs(self.log_dir, exist_ok=True)

        # Create grid directory '/Grids' as subdir of the compute directory if it doesn't exist
        self.grid_dir = f'{self.compute_dir}/Grids'
        os.makedirs(self.grid_dir, exist_ok=True)

        # Create config directory '/Configs' as subdir of compute directory if it doesn't exist
        self.config_dir = f'{self.compute_dir}/Configs'
        os.makedirs(self.config_dir, exist_ok=True)

        # Create result directory '/Results' as subdir of compute directory if it doesn't exist
        self.res_dir = f'{self.compute_dir}/Results'
        os.makedirs(self.res_dir, exist_ok=True)

        # Create global logfile 'in ComputeDir/Logs' to copy all stdout to
        self.global_logfile = f'{self.log_dir}/Global_log_{Path(self.global_config).stem}.log'
        os.makedirs(os.path.dirname(self.global_logfile), exist_ok=True)

        # Copy all future stdout to logfile
        sys.stdout = Logger(self.global_logfile)

        elapsed_dir = time.time() - start_dir
        print("Directories created. Elapsed time: {0:.3f} seconds".format(elapsed_dir))

        print(f'Compute ID: {self.compute_id}')
        print(f'Compute directory: {self.compute_dir}')
        print(f'Global logfile: {self.global_logfile}')
        print(f'Config file: {self.global_config}')

    def __del__(self):
        # Make sure to close all clients when CGS instance is destroyed
        for client in self.clients:
            client["paramiko_client"].close()

    def create_cluster(self, nodes: dict):
        # Close existing clients
        for client in self.clients:
            client["paramiko_client"].close()

        self.nodes = nodes

        ##################
        # Create cluster #
        ##################
        print("")
        print("***CREATING CLUSTER***")
        start_cluster = time.time()

        # Get password to connect to cluster nodes.
        username = getpass.getuser()
        password = getpass.getpass(prompt='Enter password: ', stream=sys.stderr)

        for node in self.nodes['hostnames']:
            client = ssh_connect(node, username=username, password=password)
            if client is not None:
                local_config_dir = f'{self.config_dir}/{node}'
                os.makedirs(local_config_dir, exist_ok=True)
                self.clients.append({
                    "paramiko_client": client,
                    "node_name": node,
                    "config_dir": local_config_dir})

        # TODO: Print/save hardware specifications of each node
        elapsed_cluster = time.time() - start_cluster
        print("Cluster created. Elapsed time: {0:.3f} seconds".format(elapsed_cluster))

        return self.clients

    def compute_grid(self, param_grid_arg, permute=True):

        #########################
        # Create parameter grid #
        #########################
        print("")
        print("***CREATING PARAMETER GRID***")
        start_grid = time.time()

        # param_grid can either be *.csv, dict or DataFrame
        if isinstance(param_grid_arg, str):
            # If parameter grid is a string
            try:
                grid = pd.read_csv(param_grid_arg)
                if 'status' not in grid.columns:
                    grid['status'] = 'unsolved'
                else:
                    # Set rows with status 'failed' to 'unsolved' to compute them again
                    unsolved_idx = grid.index[grid['status'] == "failed"]
                    grid.at[unsolved_idx, 'status'] = 'unsolved'
                if os.path.dirname(param_grid_arg) != self.grid_dir:
                    # Copy parameter grid to CGS instances' grid directory
                    param_grid_path = f'{self.grid_dir}/{os.path.basename(param_grid_arg)}'
                    grid.to_csv(param_grid_path, index=False)
                else:
                    param_grid_path = param_grid_arg
            except FileNotFoundError as err:
                print(err)
                print("Returning!")
                # Stop computation
                return
        elif isinstance(param_grid_arg, dict):
            # If parameter grid is a dictionary
            param_grid_path = f'{self.grid_dir}/DefaultGrid{self.param_grid_count}.csv'
            grid = linearize_grid(param_grid_arg, permute=permute)
            if 'status' not in grid.columns:
                grid['status'] = 'unsolved'
            else:
                # Set rows with status 'failed' to 'unsolved' to compute them again
                unsolved_idx = grid.index[grid['status'] == "failed"]
                grid.at[unsolved_idx, 'status'] = 'unsolved'
            grid.to_csv(param_grid_path, index=True)
            # To create different default grids if compute_grid() is called more than once during a computation
            self.param_grid_count += 1
        elif isinstance(param_grid_arg, pd.DataFrame):
            # If parameter grid is a pandas.DataFrame
            param_grid_path = f'{self.grid_dir}/DefaultGrid{self.param_grid_count}.csv'
            grid = param_grid_arg
            if 'status' not in grid.columns:
                grid['status'] = 'unsolved'
            else:
                # Set rows with status 'failed' to 'unsolved' to compute them again
                unsolved_idx = grid.index[grid['status'] == "failed"]
                grid.at[unsolved_idx, 'status'] = 'unsolved'
            grid.to_csv(param_grid_path, index=True)
            # To create different default grids if compute_grid() is called more than once during a computation
            self.param_grid_count += 1
        else:
            print("Parameter grid unsupported format")
            print("Returning!")
            # Stop computation
            return

        grid_name = Path(param_grid_path).stem

        elapsed_grid = time.time() - start_grid
        print("Grid created. Elapsed time: {0:.3f} seconds".format(elapsed_grid))
        print(f'Parameter grid: {param_grid_path}')
        print(grid)

        ##########################################################
        # Check parameter map and parameter grid for consistency #
        ##########################################################
        print("")
        print("***CHECKING PARAMETER GRID AND MAP FOR CONSISTENCY***")
        start_check = time.time()

        with open(self.global_config) as config:
            # Open global config file and read the config dictionary
            param_dict = json.load(config)
            try:
                # Try to read parameter map from the parameter dictionary
                param_map = param_dict['param_map']
            except KeyError as err:
                # Config_file does not contain a key 'param_map'
                print("KeyError:", err)
                print("Returning!")
                # Stop computation
                return

            if not check_key_consistency(grid, param_map):
                # Not all keys of parameter map can be found in parameter grid
                print("Not all parameter map keys found in parameter grid")
                print("Parameter map keys:")
                print(*list(param_map.keys()))
                print("Parameter grid keys:")
                print(*list(grid.keys()))
                print("Returning!")
                # Stop computation
                return
            else:
                # Parameter grid and parameter map match
                print("All parameter map keys found in parameter grid")
                grid_res_dir = f'{self.res_dir}/{grid_name}'
                # Create parameter grid specific result directory
                os.makedirs(grid_res_dir, exist_ok=True)
                # Continuing with computation

        elapsed_check = time.time() - start_check
        print("Consistency check complete. Elapsed time: {0:.3f} seconds".format(elapsed_check))

        #############################
        # Begin cluster computation #
        #############################
        print("")
        print("***BEGINNING CLUSTER COMPUTATION***")
        start_comp = time.time()

        if not self.clients:
            print("No cluster created")
            print("Please call ClusterGridSearch.create_cluster() first!")
            print("Returning")
            # Stop computation
            return
        else:
            print(f'Computing grid \'{param_grid_path}\' on nodes: ')
            for client in self.clients:
                print(client["node_name"])

        #####################
        # Start thread pool #
        #####################
        print("")
        print("***STARTING THREAD POOL***")

        # Create lock to control thread scheduling
        self.lock = RLock()

        # TODO: Implement asynchronous computation instead of multithreading

        threads = [self.__spawn_thread(client, grid, grid_name, grid_res_dir) for client in self.clients]

        # Wait for all threads to finish
        for t in threads:
            t.join()

        elapsed_comp = time.time() - start_comp

        print(grid_name)
        print("Computation finished. Elapsed time: {0:.3f} seconds".format(elapsed_comp))
        print(grid)
        print(f'Find results in: {grid_res_dir}/')
        return grid_res_dir

    def __spawn_thread(self, client, grid, grid_name, grid_res_dir):
        t = Thread(
            name=client["node_name"],
            target=self.__scheduler,
            args=(client["paramiko_client"], grid, grid_name, grid_res_dir)
        )
        t.start()
        return t

    def __scheduler(self, client, grid: pd.DataFrame, grid_name: str, grid_res_dir: str,
                    num_params=4):
        # TODO: Dynamically find the best number of params to fetch at once for each worker
        #   Maybe match with number of CPUs or memory

        thread_name = currentThread().getName()

        command = f'{self.nodes["host_env"]} {self.nodes["host_file"]}'

        # local_config_dict = dict()

        if fetch_param_idx(grid, set_status=False).isnull():
            # If param_grid has no column named 'status':
            print(f'[T]\'{thread_name}\': "No \'status\' column in param_grid')
            print(f'[T]\'{thread_name}\': Returning!')
        else:
            # TODO: if no shared directory is available:
            #   Copy environment, remote script and configs to a local directory on the host
            #   Copy results from the remote host to a local directory on the master

            # TODO: Call exec_command only once and send global config file.
            #  Send node specific config to stdin inside the while loop

            local_config_idx = 0
            local_config = f'{self.config_dir}/{thread_name}/local_config_{thread_name}_default_{local_config_idx}.csv'
            subgrid_dir = f'{self.grid_dir}/Subgrids/{thread_name}'
            os.makedirs(subgrid_dir, exist_ok=True)

            while not fetch_param_idx(grid, lock=self.lock, set_status=False).empty:

                # TODO: Read client's stdin and check if the string equals "Awaiting grid". If true, send a new
                #   parameter grid or an exit command to the host. If false, keep reading stdout.

                with self.lock:
                    start_calc = time.time()

                    # Ensure that the following code is executed without switching threads in between
                    print(f'[T]\'{thread_name}\': Fetching index: ', end="")

                    # Get indices of n parameter combinations that haven't been computed yet and create local config
                    param_idx = fetch_param_idx(grid, num_params=num_params)
                    for idx in param_idx:
                        print(f'[{idx}]', end=" ")
                    print("")

                    # Create parameter sub-grid from fetched indices to pass to the remote host
                    subgrid = f'{subgrid_dir}/Subgrid_{thread_name}_default_{local_config_idx}.csv'

                    (grid.iloc[param_idx]).to_csv(subgrid, index=True)

                    print(f'[T]\'{thread_name}\': Starting remote computation...')

                    stdin, stdout, stderr = client.exec_command(command +
                                                                f' --global_config={self.global_config}'
                                                                f' --local_config={local_config}'
                                                                f' --local_grid={subgrid}'
                                                                f' --log_dir={self.log_dir}'
                                                                f' --res_dir={grid_res_dir}'
                                                                f' --grid_name={grid_name}',
                                                                get_pty=True)

                # Wait for remote computation to finish
                # stdout.channel.recv_exit_status()

                # Print what has been sent to the channels stdout or stderr
                for line in iter(stdout.readline, ""):
                    print(f'[H]\'{thread_name}\': {line}', end="")

                # Set result status for each computed index in param_grid
                for idx in param_idx:
                    res_file = f'{grid_res_dir}/CGS_result_{grid_name}_idx_{idx}.csv'
                    try:
                        if os.path.getsize(res_file) > 0:
                            grid.at[idx, 'status'] = 'done'
                            print(f'[T]\'{thread_name}\': Computing index [{idx}]: done!')
                        else:
                            grid.at[idx, 'status'] = 'failed'
                            print(f'[T]\'{thread_name}\': Computing index [{idx}]: failed!')
                            print(f'[T]\'{thread_name}\': Resultfile {res_file} is empty')
                    except OSError as e:
                        grid.at[idx, 'status'] = 'failed'
                        print(f'[T]\'{thread_name}\': Computing index [{idx}]: failed!')
                        print(f'[T]\'{thread_name}\': Resultfile {res_file} not created')

                local_config_idx += 1

                elapsed_calc = time.time() - start_calc
                print("Elapsed time: {0:.3f} seconds".format(elapsed_calc))

            # When no more inidices can be fetched
            print(f'[T]\'{thread_name}\': No more parameter combinations available!')


def create_cgs_config(filepath, circuit_template, param_map, dt, simulation_time, inputs,
                      outputs, sampling_step_size=None, **kwargs):
    """Creates a configfile.json containing a config_dict{} with input parameters as key-value pairs

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
    kwargs

    Returns
    -------

    """
    config_dict = {
        "circuit_template": circuit_template,
        "param_map": param_map,
        "dt": dt,
        "simulation_time": simulation_time,
        "inputs": {str(*inputs.keys()): list(*inputs.values())},
        "outputs": outputs,
        "sampling_step_size": sampling_step_size,
        "kwargs": kwargs
    }

    with open(filepath, "w") as f:
        json.dump(config_dict, f, indent=2)


def ssh_connect(node, username, password):
    """Connect to a host via SSH

    Parameters
    ----------
    node
        Name or IP-address of the host to connect to
    username
    password

    Returns
    -------
    paramiko.SSHClient()
        Throws exception and returns None if connection fails. See Paramiko documentation

    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # TODO: Create connection with host-key-pair and no password

    try:
        client.connect(node, username=username, password=password)
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


def fetch_param_idx(param_grid, lock=False, num_params=1, set_status=True):
    """Fetch a pandas.Index([index_list]) with the indices of the first num_params rows of param_grid who's
    'status'-column equals 'unsolved'

    Parameters
    ----------
    param_grid
        Linearized parameter grid of type pandas.DataFrame.
    lock
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


def check_key_consistency(param_grid, param_map):
    grid_key_lst = list(param_grid.keys())
    map_key_lst = list(param_map.keys())
    return all((map_key in grid_key_lst for map_key in map_key_lst))
