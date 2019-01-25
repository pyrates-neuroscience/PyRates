
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
        self.param_grid = None
        self.param_grid_path = ""
        self.param_grid_count = 0

        #############################################################
        # Create main compute directory, subdirectories and logfile #
        #############################################################
        print("")
        print("***CREATE WORKING DIRECTORY***")

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

        print(f'Compute ID: {self.compute_id}')
        print(f'Compute directory: {self.compute_dir}')
        print(f'Global logfile: {self.global_logfile}')
        print(f'Config file: {self.global_config}')

    def __del__(self):
        # Make sure to close all clients when CGS instance is destroyed
        for client in self.clients:
            client["paramiko_client"].close()

    def create_cluster(self, nodes):
        self.nodes = nodes
        # List of directories to save local config files for each host to.

        ##################
        # Create cluster #
        ##################
        print("")
        print("***CREATING CLUSTER***")

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

    def compute_grid(self, param_grid, permute=True):

        #########################
        # Create parameter grid #
        #########################
        print("")
        print("***CREATING PARAMETER GRID***")

        # param_grid can either be *.csv, dict or DataFrame
        if isinstance(param_grid, str):
            try:
                self.param_grid = pd.read_csv(param_grid)
                if 'status' not in self.param_grid.columns:
                    self.param_grid['status'] = 'unsolved'
                else:
                    # TODO: Set status keys from 'failed' to 'unsolved' to compute them again
                    pass
                if os.path.dirname(param_grid) != self.grid_dir:
                    # Copy parameter grid to CGS instances' grid directory
                    self.param_grid_path = f'{self.grid_dir}/{os.path.basename(param_grid)}'
                    self.param_grid.to_csv(self.param_grid_path, index=False)
                else:
                    self.param_grid_path = param_grid
            except FileNotFoundError as err:
                print(err)
                print("Returning!")
                # Stop computation
                return

        elif isinstance(param_grid, dict):
            self.param_grid_path = f'{self.grid_dir}/default_grid_{self.param_grid_count}.csv'
            self.param_grid = linearize_grid(param_grid, permute=permute)
            if 'status' not in self.param_grid.columns:
                self.param_grid['status'] = 'unsolved'
            self.param_grid.to_csv(self.param_grid_path, index=False)
            # To create different default grids if compute_grid() is called more than once during a computation
            self.param_grid_count += 1

        elif isinstance(param_grid, pd.DataFrame):
            self.param_grid_path = f'{self.grid_dir}/default_grid_{self.param_grid_count}.csv'
            self.param_grid = param_grid
            if 'status' not in self.param_grid.columns:
                self.param_grid['status'] = 'unsolved'
            self.param_grid.to_csv(self.param_grid_path, index=False)
            # To create different default grids if compute_grid() is called more than once during a computation
            self.param_grid_count += 1

        else:
            print("Parameter grid unsupported format")
            print("Returning!")
            # Stop computation
            return

        print(f'Parameter grid: {self.param_grid_path}')
        print(self.param_grid)

        ##########################################################
        # Check parameter map and parameter grid for consistency #
        ##########################################################
        print("")
        print("***CHECKING PARAMETER GRID AND MAP FOR CONSISTENCY***")
        with open(self.global_config) as config:
            param_dict = json.load(config)
            try:
                param_map = param_dict['param_map']
            except KeyError as err:
                # If config_file does not contain key 'param_map'
                print("KeyError:", err)
                print("Returning!")
                # Stop computation
                return
            if not check_key_consistency(self.param_grid, param_map):
                print("Not all parameter map keys found in parameter grid")
                print("Parameter map keys:")
                print(*list(param_map.keys()))
                print("Parameter grid keys:")
                print(*list(self.param_grid.keys()))
                print("Returning!")
                # Stop computation
                return
            else:
                print("All parameter map keys found in parameter grid")
                paramgrid_respath = f'{self.res_dir}/{Path(self.param_grid_path).stem}'
                os.makedirs(paramgrid_respath, exist_ok=True)
                # Continuing with computation

        #############################
        # Begin cluster computation #
        #############################
        print("")
        print("***BEGINNING CLUSTER COMPUTATION***")

        if not self.clients:
            print("No cluster created")
            print("Please call ClusterGridSearch.create_cluster() first!")
            print("Returning")
            # Stop computation
            return
        else:
            print(f'Computing grid \'{self.param_grid_path}\' on nodes: ')
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

        threads = [self.__spawn_thread(client) for client in self.clients]

        # Wait for all threads to finish
        for t in threads:
            t.join()

        print(f'Computation finished')
        print(f'Find results in: {paramgrid_respath}')

    def __scheduler(self, client, local_config_dir):

        thread_name = currentThread().getName()

        # dummy
        # stdin, stdout, stderr = client.exec_command('ls', get_pty=True)
        #
        # for line in iter(stdout.readline, ""):
        #     print(f'[T]\'{thread_name}\': {line}', end="")
        # return

        # If param_grid has a column named 'status'
        if not fetch_param_idx(self.param_grid, set_status=False).isnull():
            command = self.nodes['host_env'] + ' ' + self.nodes['host_file']

            # TODO: Copy environment, remote script and local configs to a local directory on the host if no shared
            #  directory is available

            # TODO: Call exec_command only once and send global config file

            # TODO: Send node specific config to stdin inside a loop
            # channel.send()
            # stdin.write()
            # stdin.flush()

            local_config_dict = dict()

            while not fetch_param_idx(self.param_grid, lock=self.lock, set_status=False).empty:

                # Create local config file
                local_config_idx = 0
                local_config_path = f'{local_config_dir}/local_config_{thread_name}_default_{local_config_idx}.json'

                # Ensure that the following code is executed without switching threads in between
                with self.lock:

                    print(f'[T]\'{thread_name}\': Fetching index: ', end="")

                    # Get indices of n parameter combinations that haven't been computed yet
                    param_idx = fetch_param_idx(self.param_grid, num_params=2)
                    local_config_dict["param_idx"] = param_idx
                    print(*param_idx)

                    # Get parameter combination to pass as argument to the remote host
                    param_grid_arg = self.param_grid.iloc[param_idx]
                    local_config_dict["param_idx"] = param_idx

                    print(f'[T]\'{thread_name}\': Creating local config file: {local_config_path}')
                    json.dump(local_config_dict, local_config_path)

                    print(f'[T]\'{thread_name}\': Sending local config to remote host')

                    stdin, stdout, stderr = client.exec_command(command +
                                                                f' --global_config={self.global_config}'
                                                                f' --local_config={local_config_dir}'
                                                                f' --log_path={self.log_dir}'
                                                                f' --result_path={self.res_dir}'
                                                                f' --grid_name={Path(self.param_grid_path).stem}',
                                                                get_pty=True)

                # Wait for remote computation to finish
                stdout.channel.recv_exit_status()

                # Print what has been sent to the channels stdout or stderr
                # for line in iter(stdout.readline, ""):
                #     print(f'[H]\'{thread_name}\': {line}', end="")

                # TODO: Check if resultfile has been created and is not empty. If so, change status of current
                #  param_idx in param_grid from 'pending' to 'done'. Otherwise set status to 'failed'
                # resultfile = f'CGS_result_{grid_name}_idx_{param_idx[n]}.csv'
        else:
            # If no column named 'status' in param_grid:
            print(f'[T]\'{thread_name}\': "No column named \'status\' in param_grid')

        # TODO: If no shared memory is available, copy result files from host back to local workstation

    def __spawn_thread(self, client):
        t = Thread(
            name=client["node_name"],
            target=self.__scheduler,
            args=(client["paramiko_client"], client["config_dir"])
        )
        t.start()
        return t

    def __set_param_result_status(self):
        pass


def create_cgs_config(filepath, circuit_template, param_map, dt, simulation_time, inputs,
                      outputs, sampling_step_size=None, **kwargs):
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
