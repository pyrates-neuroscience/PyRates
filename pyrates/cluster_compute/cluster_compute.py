
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
# system imports
import os
import sys
import json
import time as t
import glob
import getpass
import h5py
from shutil import copy2
from pathlib import Path
from datetime import datetime
from threading import Thread, currentThread, RLock

# external imports
import pandas as pd
import numpy as np
import paramiko

# meta infos
__author__ = "Christoph Salomon"
__status__ = "development"


class StreamTee(object):
    # TODO: Stop stream tee after cluster computation has finished

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
        Creates a logfile in ComputeDirectory/Logs and tees all future stdout and stderr of the executing script
        to this file
        Connects to all nodes via SSH and saves the corresponding paramiko client

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

    def __del__(self):
        # Make sure to close all clients when cluster_examples instance is destroyed
        for client in self.clients:
            client["paramiko_client"].close()

    def cluster_connect(self, nodes):
        """Connect to all nodes via SSH

        Connect to all nodes in the given list via SSH, using ssh_connect()
        Adds a dictionary for each node to the class internal 'clients' list
        Each dictionary contains:
            - ["paramiko_client"]: A paramiko client that can be used to execute commands on the node
            - ["node_name"]: The name of the connected node
            - ["hardware"]: A dictionary with certain hardware information of the node
            - ["logfile"]: Path to a logfile inside the instance's working directory,
                           where all stdout and stderr of the node will be redirected to

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
                # Create local logfiles
                local_logfile = f'{self.log_dir}/Local_logfile_{node}.log'
                os.makedirs(os.path.dirname(local_logfile), exist_ok=True)

                hw = self.get_hardware_spec(client)

                self.clients.append({
                    "paramiko_client": client,
                    "node_name": node,
                    "hardware": hw,
                    "logfile": local_logfile})
            print("")

    def run(self, thread_kwargs: dict, **kwargs):
        """Start a thread for each connected client. Each thread executes the thread_master() function

        Each thread and therefor each instance of the thread_master() function is responsible for the communication
        with one worker node in the cluster.
        Stops execution of the outside script until all threads have finished.
        Can be called multiple times from the same ClusterCompute instance

        Params
        ------
        kwargs

        Returns
        -------

        """

        t0 = t.time()

        # ! Insert preprocessing of input data can here !

        # **kwargs of spawn_thread() will be parsed as dict to thread_master()
        threads = [self.spawn_thread(client, thread_kwargs) for client in self.clients]
        for t_ in threads:
            t_.join()

        print("")
        print(f'Cluster computation finished. Elapsed time: {t.time()-t0:.3f} seconds')

    def spawn_thread(self, client, thread_kwargs):
        t_ = Thread(
            name=client["node_name"],
            target=self.thread_master,
            args=(client, thread_kwargs)
        )
        t_.start()
        return t_

    def thread_master(self, client, thread_kwargs_: dict):
        """Function that is executed by every thread. Every instance of thread_master is bound to a different client

        The thread_master() function can be arbitrarily changed to fit the needs of the user.
        Commandline commands on the remote node can be executed using client["paramiko_client"].exec_command("")
        self.lock can be used to ensure that a code snippet is executed without threads being switched in between
        e.g.
            with self.lock:
                some code that is executed without switching to another thread
        Since thread_master() is called by spawn_threads(), which again is called by run(), **kwargs of
        spawn_threads() are parsed as a single dict to thread_master(). **kwargs of run() are NOT automatically parsed
        to spawn_threads() or thread_master().

        Params
        ------
        client
            dict containing a paramiko client, name of the connected node, dict with hardware specifications and
            a path to a logfile.
        kwargs_
            dict containing **kwargs of spawn_threads() as key/value pairs

        Returns
        -------

        """

        # Add funcionality here

        pass

    @staticmethod
    def get_hardware_spec(pm_client):
        """Print CPU model, number of CPU cores, min CPU freq, max CPU freq and working memory of a paramiko client
        connected to a remote worker node

        Parameters
        ----------
        pm_client
            Paramiko client

        Returns
        -------
            dict containing the above mentioned hardware specifications as key/value pairs

        """

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
        """Connect to a host via SSH an return a respective paramiko.SSHClient

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

    def run(self, thread_kwargs: dict, **kwargs):
        """

        :param thread_kwargs:
        :param kwargs:
        :return:
        """
        t_total = t.time()

        # Unzip kwargs
        config_file = kwargs["config_file"]
        param_grid_temp = kwargs["param_grid"]
        permute = kwargs["permute"]
        chunk_size = kwargs["chunk_size"]

        print("")

        print("***PREPARING PARAMETER GRID***")
        t0 = t.time()
        param_grid, grid_file, grid_name = self.prepare_grid(param_grid_temp, permute)
        print(f'Done. Elapsed time: {t.time()-t0:.3f} seconds')

        print("")

        print("***CHECKING PARAMETER GRID AND MAP FOR CONSISTENCY***")
        t0 = t.time()
        # Check parameter map and parameter grid for consistency
        with open(config_file) as config:
            # Open global config file and read the config dictionary
            param_dict = json.load(config)
            try:
                # Read parameter map from the parameter dictionary
                param_map = param_dict['param_map']
            except KeyError as err:
                # Config_file does not contain a key 'param_map'
                print("KeyError:", err)
                print("Returning!")
                # Stop computation
                return

            if not self.check_key_consistency(param_grid, param_map):
                # Not all keys of parameter map can be found in parameter grid
                print("Not all parameter map keys found in parameter grid")
                print("Parameter map keys:")
                print(*list(param_map.keys()))
                print("Parameter grid keys:")
                print(*list(param_grid.keys()))
                print("Returning!")
                # Stop computation
                return
            else:
                # Parameter grid and parameter map match
                print("All parameter map keys found in parameter grid")
                # Continuing with computation

        # Create parameter grid specific result directory and result file
        grid_res_dir = f'{self.res_dir}/{grid_name}'
        os.makedirs(grid_res_dir, exist_ok=True)
        grid_res_file = f'{grid_res_dir}/CGS_result_{grid_name}.h5'
        print(f'Elapsed time: {t.time()-t0:.3f} seconds')

        print("")

        print("***PREPARING CHUNK SIZES***")
        t0 = t.time()
        # Prepare the chunk sizes
        self.prepare_chunks(chunk_size, len(param_grid))
        print(f'Done. Elapsed time: {t.time() - t0:.3f} seconds')

        print("")

        # Update thread keywords dictionary
        thread_kwargs["param_grid"] = param_grid
        thread_kwargs["grid_name"] = grid_name
        thread_kwargs["grid_res_dir"] = grid_res_dir
        thread_kwargs["config_file"] = config_file
        thread_kwargs["global_res_file"] = grid_res_file

        print("***STARTING THREAD POOL***")
        # Spawn threads to control the node connections
        threads = [self.spawn_thread(client, thread_kwargs) for client in self.clients]
        for t_ in threads:
            t_.join()

        print("")
        print(f'Cluster computation finished. Elapsed time: {t.time() - t_total:.3f} seconds')
        print(f'Find results in: {grid_res_dir}/')
        print("")
        param_grid.to_csv(f'{os.path.dirname(grid_file)}/{grid_name}_{self.compute_id}_ResultStatus.csv', index=True)

        return grid_res_file, grid_file

    def thread_master(self, client, thread_kwargs: dict):
        thread_name = currentThread().getName()

        # Get client information
        pm_client = client["paramiko_client"]
        logfile = client["logfile"]
        num_params = client["num_params"]

        # Get kwargs
        config_file = thread_kwargs["config_file"]
        param_grid = thread_kwargs["param_grid"]
        grid_name = thread_kwargs["grid_name"]
        grid_res_dir = thread_kwargs["grid_res_dir"]
        worker_env = thread_kwargs["worker_env"]
        worker_file = thread_kwargs["worker_file"]
        global_res_file = thread_kwargs["global_res_file"]

        command = f'{worker_env} {worker_file}'

        # Create folder to save local subgrids to
        subgrid_dir = f'{self.grid_dir}/Subgrids/{thread_name}'
        os.makedirs(subgrid_dir, exist_ok=True)
        subgrid_idx = 0

        while not self.fetch_param_idx(param_grid, lock=self.lock, set_status=False).empty:

            with self.lock:
                # Fetch grid indices
                ####################
                param_idx = self.fetch_param_idx(param_grid, num_params=num_params)
                print(f'[T]\'{thread_name}\': Fetching {len(param_idx)} indices: ', end="")
                print(f'[{param_idx[0]}] - [{param_idx[-1]}]')

                # Create parameter sub-grid
                ###########################
                subgrid_fp = f'{subgrid_dir}/{thread_name}_{grid_name}_Subgrid_{subgrid_idx}.h5'
                subgrid_df = param_grid.iloc[param_idx]
                subgrid_df.to_hdf(subgrid_fp, key='Data')
                subgrid_idx += 1

                # Create temporary result file for current subgrid
                ##################################################
                idx_min = np.amin(param_idx.values)
                idx_max = np.amax(param_idx.values)
                res_file = f'{grid_res_dir}/CGS_result_{grid_name}_idx_{idx_min}-{idx_max}_temp.h5'

                # Execute worker script on the remote host
                ##########################################
                t0 = t.time()
                print(f'[T]\'{thread_name}\': Starting remote computation...')

                stdin, stdout, stderr = pm_client.exec_command(command +
                                                               f' --config_file={config_file}'
                                                               f' --subgrid={subgrid_fp}'
                                                               f' --res_file={res_file}'
                                                               f' &>> {logfile}',
                                                               get_pty=True)

            # Wait for remote computation to finish
            #######################################
            stdout.channel.recv_exit_status()
            print(f'[T]\'{thread_name}\': Remote computation finished. Elapsed time: {t.time()-t0:.3f} seconds')

            # Write results from temp to global result file
            ###############################################
            t0 = t.time()
            with self.lock:
                with h5py.File(global_res_file, 'a') as fd:
                    if 'GridIndex/' not in fd.keys():
                        group = fd.create_group(f'GridIndex/')
                    else:
                        group = fd['GridIndex']
                    with h5py.File(res_file, 'r') as fs:
                        for index_key in list(fs['GridIndex'].keys()):
                            fs.copy(f'GridIndex/{index_key}/', group)
                    os.remove(res_file)
            print(f'[T]\'{thread_name}\': Result file created. Elapsed time: {t.time()-t0:.3f} seconds')

            # Update parameter grid status flags
            ####################################
            print(f'[T]\'{thread_name}\': Updating grid status')
            for idx in param_idx:
                res_file = f'{grid_res_dir}/CGS_result_{grid_name}_idx_{idx}.h5'
                param_grid.at[idx, 'worker'] = thread_name
                try:
                    if os.path.getsize(res_file) > 0:
                        param_grid.at[idx, 'status'] = 'done'
                    else:
                        param_grid.at[idx, 'status'] = 'failed'
                except OSError as e:
                    param_grid.at[idx, 'status'] = 'failed'

        # End of while loop
        print(f'[T]\'{thread_name}\': No more parameter combinations available!')

    # Helper functions, ClusterGridSearch only
    def prepare_grid(self, param_grid_arg, permute=False):
        """Create a DataFrame and a DefaultGrid.csv from a *.csv/DataFrame/dict containing all parameter combinations

        Parameters
        ---------
        param_grid_arg
            Can be either a csv-file, a pandas DataFrame or a dictionary
            If a csv-file is given, a DataFrame is created and the csv-file is copied to the project folder
            If a DataFrame is given, a default csv-file is created in the project folder. Existing default grid files
                in the project folder are NOT overwritten. Each grid file gets it's own index
            If a dictionary is given, a DataFrame and a csv-file are created
        permute
            Only relevant when param_grid_arg is a dictionary.
            If true, a DataFrame is created containing all permutations of the parameters given in the dictionary as
                lists

        Returns
        -------
        grid
            Pandas DataFrame containing all parameter combinations to be computed in the cluster
        grid_file
            path to a csv.file, containing all parameter combinations of the DataFrame

        """
        # param_grid can either be *.csv, dict or DataFrame
        if isinstance(param_grid_arg, str):
            # If parameter grid is a string
            try:
                # Open grid file
                grid = pd.read_csv(param_grid_arg)
                # Check directory of grid file
                if os.path.dirname(param_grid_arg) != self.grid_dir:
                    # Copy parameter grid to cluster_examples instances' grid directory
                    copy2(param_grid_arg, self.grid_dir)
                    grid_file = f'{self.grid_dir}/{os.path.basename(param_grid_arg)}'
                else:
                    grid_file = param_grid_arg
                # Add status column
                if 'status' not in grid.columns:
                    # Add status-column
                    grid['status'] = 'unsolved'
                else:
                    # Set rows with status 'failed' to 'unsolved' to compute them again
                    unsolved_idx = grid.index[grid['status'] == "failed"]
                    grid.at[unsolved_idx, 'status'] = 'unsolved'
                # Add/reset worker-column
                grid['worker'] = ""
                return grid, grid_file
            except FileNotFoundError as err:
                print(err)
                print("Returning!")
                # Stop computation
                return None

        elif isinstance(param_grid_arg, dict):
            # Create DataFrame from dict
            grid_frame = linearize_grid(param_grid_arg, permute=permute)
            # Create default parameter grid csv-file
            grid_idx = 0
            # grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'
            grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.h5'
            # If grid_file already exist
            while os.path.exists(grid_file):
                grid_idx += 1
                grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.h5'
            # grid_frame.to_csv(grid_file, index=True, float_format='%g')
            # grid_frame.to_csv(grid_file, index=True)
            grid_frame.to_hdf(grid_file, key='Data')
            # Add status columns to grid
            if 'status' not in grid_frame.columns:
                grid_frame['status'] = 'unsolved'
            else:
                # Set rows with status 'failed' to 'unsolved' to compute them again
                unsolved_idx = grid_frame.index[grid_frame['status'] == "failed"]
                grid_frame.at[unsolved_idx, 'status'] = 'unsolved'
            # Add/reset worker-column
            grid_frame['worker'] = ""
            grid_name = Path(grid_file).stem
            return grid_frame, grid_file, grid_name

        elif isinstance(param_grid_arg, pd.DataFrame):
            grid_frame = param_grid_arg
            # Create default parameter grid csv-file from DataFrame
            grid_idx = 0
            grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'
            # If grid_file already exist
            while os.path.exists(grid_file):
                grid_idx += 1
                grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'
            grid_frame.to_csv(grid_file, index=True)
            # Add status columns to grid
            if 'status' not in grid_frame.columns:
                grid_frame['status'] = 'unsolved'
            else:
                # Set rows with status 'failed' to 'unsolved' to compute them again
                unsolved_idx = grid_frame.index[grid_frame['status'] == "failed"]
                grid_frame.at[unsolved_idx, 'status'] = 'unsolved'
            # Add/reset worker-column
            grid_frame['worker'] = ""
            grid_name = Path(grid_file).stem
            return grid_frame, grid_file, grid_name

        else:
            print("Parameter grid unsupported format")
            # Stop computation
            return None

    def prepare_chunks(self, chunk_size, grid_len):
        """Set the amount of parameter combinations that are computed at once by each worker

        Adds a key "num_params" to each client dict in self.clients

        Parameters
        ----------
        chunk_size
            int or str containing either a fixed chunk size for each worker or a mode to compute the chunk size
            dynamically. Mode can be one out of:
            'dist_equal': The number of parameters is equally distributed among the workers.
                The first node to finish it's computation starts another one to compute the remaining parameters
            'dist_equal_add_mod': The number of parameters is equally distributed among the workers.
                The remaining amount of parameters (modulo) is added to the chunk size of the first node in the
                node list.
            'fit_hardware': Not implemented yet
        grid_len
            Total number of parameter combinations computed by all workers together

        Returns
        ------

        """

        if isinstance(chunk_size, int):
            num_params = chunk_size
            for client in self.clients:
                client['num_params'] = num_params
        elif isinstance(chunk_size, str):
            if chunk_size == "dist_equal_add_mod":
                # grid_len = len(grid_frame)
                num_clients = len(self.clients)
                num_params = int(grid_len/num_clients)
                mod = grid_len % num_clients
                # Distribute all parameters equally among all nodes
                for client in self.clients:
                    # Add remaining parameters to the first node
                    if mod != 0:
                        client['num_params'] = num_params + mod
                        mod = 0
                    else:
                        client['num_params'] = num_params
            elif chunk_size == "dist_equal":
                num_clients = len(self.clients)
                num_params = int(grid_len/num_clients)
                # Distribute all parameters equally among all nodes
                for client in self.clients:
                    client['num_params'] = num_params
            elif chunk_size == "fit_hardware":
                pass
        else:
            print("num_params: Unsupported command. Returning")
            return None

    @staticmethod
    def check_key_consistency(param_grid, param_map):
        """
        Parameters
        ----------
        param_grid:
        param_map:

        Returns
        -------

        """
        grid_key_lst = list(param_grid.keys())
        map_key_lst = list(param_map.keys())
        return all((map_key in grid_key_lst for map_key in map_key_lst))

    @staticmethod
    def fetch_param_idx(param_grid, lock=None, num_params=1, set_status=True):
        """Fetch indices of the first num_params rows of param_grid that's status-column equals 'unsolved'

        Parameters
        ----------
        param_grid
            Linearized parameter grid of type pandas.DataFrame.
        lock
            RLock to make sure threads are not switched during the execution of this function
        num_params
            Number of indices to fetch from param_grid. Is 1 by default.
        set_status
            If True, sets 'status' key of the fetched rows to 'pending', to exclude them from future calls.
            Can be used to check param_grid for fetchable or existent keys without changing their 'status' key if False.
            Is True by default.

        Returns
        -------
        pandas.Index([index_list])
            Is empty if there are no row indices to be fetched.
            Is np.nan if param_grid has no key named 'status'.
            Contains all remaining indices if num_params is higher than row indices left.

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


class ClusterBenchmark(ClusterCompute):
    pass


# Utility functions
def create_cgs_config(fp, circuit_template, param_map, dt, simulation_time, inputs,
                      outputs, sampling_step_size=None, **kwargs):
    """Creates a configfile.json containing a config_dict{} with all input parameters as key-value pairs

    Parameters
    ----------
    fp
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
    if not os.path.exists(fp):
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
        with open(fp, "w") as f:
            json.dump(config_dict, f, indent=2)
    else:
        print(f'Configfile: {fp} already exists.')


def read_cgs_results(res_file, key='Data', filter_grid=None):
    """ Collect data from all csv-files in the res_dir inside on DataFrame

    Parameters
    ----------
    res_file
        Directory with csv-files
    data_key
    filter_grid
        Not implemented yet

    Returns
    -------
    DataFrame

    """
    list_ = []
    with(h5py.File(res_file, 'r')) as file_:
        keys = list(file_['GridIndex'].keys())

    for i, index_key in enumerate(keys):
        df = pd.read_hdf(res_file, key=f'/GridIndex/{index_key}/{key}/')
        list_.append(df)

    return pd.concat(list_, axis=1)


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


def create_resultfile(fp_res, fp_h5, delete_old=False):
    """

    :param fp_res:
    :param fp_h5:
    :param delete_old:
    :return:
    """
    """Create one hdf5-file from multiple hdf5-files"""
    files = glob.glob(fp_res + "/*.h5")
    with pd.HDFStore(fp_h5, "w") as store:
        for i, file_ in enumerate(files):
            df = pd.read_hdf(file_, key='Data')
            idx = Path(file_).stem.rsplit('_', 1)[-1]
            store.put(key=f'/Data/Idx_{idx}/', value=df)
            if delete_old:
                os.remove(file_)
#     files = glob.glob(fp_res + "/*.h5")
#     with h5py.File(fp_h5, "w") as f:
#         for i, file_ in enumerate(files):
#             df = pd.read_hdf(file_, key='Data')
#             f.create_dataset(name=f'/Data/Idx_{i}/Values/', data=df.values)
#             f.create_dataset(name=f'/Data/Idx_{i}/Columns/', data=df.columns.to_frame())
#             if i == 0:
#                 f.create_dataset(name=f'Index', data=df.index)
#             if delete_temp:
#                 os.remove(file_)


def plot_frame(data):
    import matplotlib.pyplot as plt
    from seaborn import cubehelix_palette
    from pyrates.utility import plot_connectivity

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})

    cm1 = cubehelix_palette(n_colors=int(data.size), as_cmap=True, start=2.5, rot=-0.1)

    cax1 = plot_connectivity(data.values, ax=ax, yticklabels=list(np.round(data.index, decimals=2)),
                             xticklabels=np.arange(len(data.columns.values)), cmap=cm1)
    cax1.set_xlabel('Grid index')
    cax1.set_ylabel('Values')
    cax1.set_title(f'Data frame plot')
    plt.show()
