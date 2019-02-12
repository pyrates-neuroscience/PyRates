
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
# TODO: Why does it take so long to start the master/worker script? PyRates?
# system imports
import os
import sys
import json
import time
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


class ClusterGridSearch(object):
    def __init__(self, global_config, compute_dir=None, **kwargs):
        """Create new ClusterGridSearch instance

        Each ClusterGridSearch instance has its own unique compute ID, which is assigned when a new CGS instance is
            created.
        When a new CGS instance is invoked, a folder with all necessary directories is created in the directory of the
            global config file if not specified otherwise
        The global config file is copied to the instances compute directory
        All prints to stdout and stderr are copied to a global logfile, that is created during the initialization

        Parameters
        ----------
        global_config:
            JSON file with all necessary information to compute a grid with PyRate's grid_search method, except for a
            parameter grid, which is passed later
        compute_dir:
            If given, all necessary subfolders for the computation are created inside the specified directory. If no
            compute directory is given, a default directory is created based on the location of the global config file
            and the unique compute ID of the CGS instance
        kwargs:
        """
        self.global_config = global_config

        # self members changed outside the __init__
        self.node_config = None
        self.clients = []
        self.lock = None

        #############################################################
        # Create main compute directory, subdirectories and logfile #
        #############################################################
        print("")
        print("***CREATE WORKING DIRECTORY***")
        # start_dir = time.time()

        # Unique id of current ClusterGridSearch instance.
        self.compute_id = datetime.now().strftime("%d%m%y-%H%M%S")

        # Name of current computation. Used for unique log file and default compute directory if none is given
        self.compute_name = f'{Path(self.global_config).stem}_{self.compute_id}'

        if compute_dir:
            # Use given directory as main compute directory
            self.compute_dir = compute_dir
            os.makedirs(self.compute_dir, exist_ok=True)
        else:
            # Create default compute directory as subdir of global_config location
            self.compute_dir = f'{os.path.dirname(self.global_config)}/{self.compute_name}'
            os.makedirs(self.compute_dir, exist_ok=True)

        # Create log directory '/Logs' as subdir of compute directory
        self.log_dir = f'{self.compute_dir}/Logs/{self.compute_id}'
        os.makedirs(self.log_dir, exist_ok=True)

        # Create grid directory '/Grids' as subdir of compute directory
        self.grid_dir = f'{self.compute_dir}/Grids'
        os.makedirs(self.grid_dir, exist_ok=True)

        # Create config directory '/Config' as subdir of compute directory
        self.config_dir = f'{self.compute_dir}/Config'
        os.makedirs(self.config_dir, exist_ok=True)

        # Create result directory '/Results' as subdir of compute directory
        self.res_dir = f'{self.compute_dir}/Results'
        os.makedirs(self.res_dir, exist_ok=True)

        # Create global logfile in 'ComputeDir/Logs' to copy all stdout to
        self.global_logfile = f'{self.log_dir}/Global_log_{Path(self.global_config).stem}.log'
        os.makedirs(os.path.dirname(self.global_logfile), exist_ok=True)

        # Copy all stdout and stderr to logfile
        sys.stdout = StreamTee(sys.stdout, self.global_logfile)
        sys.stderr = StreamTee(sys.stderr, self.global_logfile)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        # elapsed_dir = time.time() - start_dir
        # print("Directories created. Elapsed time: {0:.3f} seconds".format(elapsed_dir))

        print(f'Compute ID: {self.compute_id}')
        print(f'Compute directory: {self.compute_dir}')
        print(f'Config file: {self.global_config}')
        print(f'Global logfile: {self.global_logfile}')
        print("")

        # Copy global config file to config folder
        print(f'Copying config file to compute directory...')
        copy2(global_config, self.config_dir)

    def __del__(self):
        # Make sure to close all clients when CGS instance is destroyed
        for client in self.clients:
            client["paramiko_client"].close()

    def create_cluster(self, node_config: dict):
        """Create a new compute cluster for the CGS instance

        Connects to all hosts given in the nodes dictionary using ssh_connect() and adds the corresponding client to
            the CGS internal clients list.
        Each client can be used to execute command-line commands on the connected remote machine.
        The internal client list holds a dictionary for each client, containing:
            - the paramiko client to execute commands
            - the name of the remote host
            - a directory to place node specific config files in
            - a dictionary with certain hardware information of the remote workstation

        Parameters
        ----------
        node_config
            Dictionary containing the following entries:
            - 'hostnames': List of computer names in the local network to connect to via SSH
            - 'host_env_cpu': Path to a python executable that runs the remote worker python script
                              Needs to be inside a virtual env that contains all necessary packages
            - 'host_file': Full path to the python script that is going to be executed on the remote worker

        Returns
        -------
        list of client dictionaries
            For information about the content of the dictionaries see detailed documentation above

        """
        # Close existing clients and delete them from clients list
        for client in self.clients:
            client["paramiko_client"].close()
        self.clients.clear()

        # self.nodes = nodes
        self.node_config = node_config

        ##################
        # Create cluster #
        ##################
        print("")
        print("***CREATING CLUSTER***")
        start_cluster = time.time()

        username = getpass.getuser()
        # Password request if authentication via Kerberos is not available in your cluster
        # password = getpass.getpass(prompt='Enter password: ', stream=sys.stderr)

        for node in self.node_config['hostnames']:
            client = self.__ssh_connect(node, username=username)
            if client is not None:
                hw = self.__get_hardware_spec(client)
                self.clients.append({
                    "paramiko_client": client,
                    "node_name": node,
                    "hardware": hw})

            print("")

        elapsed_cluster = time.time() - start_cluster
        print("Done! Elapsed time: {0:.3f} seconds".format(elapsed_cluster))

        return self.clients

    def compute_grid(self, param_grid_arg, num_params="dist_equal_add_mod", permute=False):
        """Compute the circuit for each parameter combination in the parameter grid utilizing a compute cluster

        Can only run when a compute-cluster for the CGS instance has been created before.
        If the parameter grid is given as a csv-file, the file is copied to folder '/Grids/' inside the CGS instance's
            working directory.
        If the parameter grid is given as a pandas.DataFrame or a dictionary, a csv-file with a default name is created
            in the CGS instance's '/Grids/' folder.
        For each call of compute_grid within the same CGS instance, a different default grid is created.
        Checks the parameter map of the CGS instance and the parameter grid for consistency. Each key in the parameter
            map has to be declared in the parameter grid
        Creates a single result file for each parameter combination in the parameter grid.
        All results files are saved to '/CGSWorkingDirectory/Results/name_of_grid/'
        Each result file contains the name of the parameter grid it belongs to and the respective index of the parameter
            combination which was used to compute the the results.
            e.g. /CGS_WorkingDir/Results/DefaultGrid0/CGS_result_DefaultGrid_0_idx_4.csv
        The config file of the CGS instance can be found in /CGS_WorkingDir/Configs/

        Parameters
        ----------
        param_grid_arg
            dict, DataFrame or csv-file with all parameter combinations to run the circuit with
        num_params
            Amount of parameter combinations, that are executed at once by one worker
            Can be a fixed number or one of the following modes:
            'dist_equal': The number of parameters is equally distributed among the workers.
                The first node to finish it's computation starts another one to compute the remaining parameters
                (modulo)
            'dist_equal_add_mod': The number of parameters is equally distributed among the workers.
                The remaining amount of parameters (modulo) is added to the chunk size of the first node in the
                node list.
            'fit_hardware': Not implemented yet
        permute
            If param_grid is a dict containing lists, a DataFrame is created with all permutations of the list entries

        Returns
        -------
        res_dir
            Folder containing all result files for the given parameter grid
        grid_file
            csv-file containing all parameter combinations used to compute the circuit

        """
        ############################
        # Preparing parameter grid #
        ############################
        print("")
        print("***PREPARING PARAMETER GRID***")
        start_grid = time.time()

        grid, grid_file = self.__prepare_grid(param_grid_arg, permute)
        if grid is None:
            print("Unable to pre process parameter grid!")
            print("Returning!")
            return

        grid_name = Path(grid_file).stem

        elapsed_grid = time.time() - start_grid
        print(f'Done. Elapsed time: {elapsed_grid:.3f} seconds')

        ##########################################################
        # Check parameter map and parameter grid for consistency #
        ##########################################################
        print("")
        print("***CHECKING PARAMETER GRID AND MAP FOR CONSISTENCY***")
        start_check = time.time()

        # Check parameter map and parameter grid for consistency
        with open(self.global_config) as config:
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

        ###################################
        # Set grid chunks for each worker #
        ###################################
        if isinstance(num_params, int):
            num_params_arg = num_params
            for client in self.clients:
                client['num_params'] = num_params_arg
        elif isinstance(num_params, str):
            if num_params == "dist_equal_add_mod":
                grid_len = len(grid)
                num_clients = len(self.clients)
                num_params_arg = int(grid_len/num_clients)
                mod = grid_len % num_clients
                # Distribute all parameters equally among all nodes
                for client in self.clients:
                    # Add remaining parameters to the first node
                    if mod != 0:
                        client['num_params'] = num_params_arg + mod
                        mod = 0
                    else:
                        client['num_params'] = num_params_arg
            elif num_params == "dist_equal":
                grid_len = len(grid)
                num_clients = len(self.clients)
                num_params_arg = int(grid_len/num_clients)
                # Distribute all parameters equally among all nodes
                for client in self.clients:
                    client['num_params'] = num_params_arg
            elif num_params == "fit_hardware":
                pass
        else:
            print("num_params: Unsupported command. Returning")
            return None

        #############################
        # Begin cluster computation #
        #############################
        print("")
        print("***BEGINNING CLUSTER COMPUTATION***")
        start_comp = time.time()

        # Check if cluster was created
        if not self.clients:
            print("No cluster created")
            print("Please call create_cluster() first!")
            print("Returning")
            # Stop computation
            return
        else:
            print(f'Computing grid \'{grid_file}\' on nodes: ')
            for client in self.clients:
                print(client["node_name"])

        #####################
        # Start thread pool #
        #####################
        print("")
        print("***STARTING THREAD POOL***")

        # Create lock to control thread scheduling
        self.lock = RLock()

        # TODO: Implement asynchronous computation instead of multithreading?

        # Spawn a thread for each client to control its execution
        threads = [self.__spawn_thread(client, grid, grid_name, grid_res_dir) for client in self.clients]

        # Wait for all threads to finish
        for t in threads:
            t.join()

        elapsed_comp = time.time() - start_comp
        print("")
        print("Computation finished. Elapsed time: {0:.3f} seconds".format(elapsed_comp))
        print("")

        # print("***PARAMETER GRID STATUS***")
        # print(grid)
        # print("")

        print(f'Find results in: {grid_res_dir}/')
        print("")
        grid.to_csv(f'{os.path.dirname(grid_file)}/{grid_name}_{self.compute_id}_ResultStatus.csv', index=True)

        return grid_res_dir, grid_file

    def __prepare_grid(self, param_grid_arg, permute=False):
        """Create a pandas.DataFrame and a default .csv file from a given set of parameter combinations

        Parameters
        ---------
        param_grid_arg
            Can be either a path to a csv-file, a pandas DataFrame or a dictionary
            If a csv-file is given, a DataFrame is created and the csv-file is copied to the project folder
            If a DataFrame is given, a default csv-file is created in the project folder. Existing default grid files
                in the project folder are NOT overwritten. Each default file gets an index so no name conflicts occur
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
                    # Copy parameter grid to CGS instances' grid directory
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
            grid = linearize_grid(param_grid_arg, permute=permute)
            # Create default parameter grid csv-file
            grid_idx = 0
            grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'
            # If grid_file already exist
            while os.path.exists(grid_file):
                grid_idx += 1
                grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'
            grid.to_csv(grid_file, index=True)
            # Add status columns to grid
            if 'status' not in grid.columns:
                grid['status'] = 'unsolved'
            else:
                # Set rows with status 'failed' to 'unsolved' to compute them again
                unsolved_idx = grid.index[grid['status'] == "failed"]
                grid.at[unsolved_idx, 'status'] = 'unsolved'
            # Add/reset worker-column
            grid['worker'] = ""
            return grid, grid_file

        elif isinstance(param_grid_arg, pd.DataFrame):
            grid = param_grid_arg
            # Create default parameter grid csv-file from DataFrame
            grid_idx = 0
            grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'
            # If grid_file already exist
            while os.path.exists(grid_file):
                grid_idx += 1
                grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'
            grid.to_csv(grid_file, index=True)
            # Add status columns to grid
            if 'status' not in grid.columns:
                grid['status'] = 'unsolved'
            else:
                # Set rows with status 'failed' to 'unsolved' to compute them again
                unsolved_idx = grid.index[grid['status'] == "failed"]
                grid.at[unsolved_idx, 'status'] = 'unsolved'
            # Add/reset worker-column
            grid['worker'] = ""
            return grid, grid_file

        else:
            print("Parameter grid unsupported format")
            # Stop computation
            return None

    def __spawn_thread(self, client: dict, grid: pd.DataFrame, grid_name: str, grid_res_dir: str):
        """Spawn a thread to control a remote cluster worker

        Parameters
        ----------
        client
            Dictionary containing
        grid
        grid_name
        grid_res_dir

        Returns
        -------
        threading.Thread

        """
        t = Thread(
            name=client["node_name"],
            target=self.__scheduler,
            args=(client["paramiko_client"], grid, grid_name, grid_res_dir, client["num_params"])
        )
        t.start()
        return t

    def __scheduler(self, client, grid: pd.DataFrame, grid_name: str, grid_res_dir: str,
                    num_params, gpu=False):

        thread_name = currentThread().getName()

        # Create logfile to redirect all stdout and stderr of the worker
        logfile = f'{self.log_dir}/Local_log_{Path(self.global_config).stem}_{thread_name}.log'
        os.makedirs(os.path.dirname(logfile), exist_ok=True)

        # Create folder to save subgrids to
        subgrid_dir = f'{self.grid_dir}/Subgrids/{thread_name}'
        os.makedirs(subgrid_dir, exist_ok=True)
        subgrid_idx = 0

        # Get python executable and path to worker script
        if gpu:
            command = f'{self.node_config["host_env_gpu"]} {self.node_config["host_file"]}'
        else:
            command = f'{self.node_config["host_env_cpu"]} {self.node_config["host_file"]}'

        while not self.__fetch_param_idx(grid, lock=self.lock, set_status=False).empty:
            # Ensure that the following code is executed without switching threads
            with self.lock:
                # Fetch grid indices
                param_idx = self.__fetch_param_idx(grid, num_params=num_params)
                print(f'[T]\'{thread_name}\': Fetching {len(param_idx)} indices: ', end="")
                print(f'[{param_idx[0]}] - [{param_idx[-1]}]')

                # Create parameter sub-grid from fetched indices to pass to the remote host
                subgrid = f'{subgrid_dir}/{thread_name}_{grid_name}_Subgrid_{subgrid_idx}.csv'
                subgrid_frame = grid.iloc[param_idx]
                subgrid_frame.to_csv(subgrid, index=True)
                subgrid_idx += 1

                # Execute worker script on the remote host
                print(f'[T]\'{thread_name}\': Starting remote computation...')
                start_calc = time.time()

                stdin, stdout, stderr = client.exec_command(command +
                                                            f' --global_config={self.global_config}'
                                                            f' --subgrid={subgrid}'
                                                            f' --res_dir={grid_res_dir}'
                                                            f' --grid_name={grid_name}'
                                                            f' &>> {logfile}',
                                                            get_pty=True)

            # Lock released. Wait for remote computation to finish
            stdout.channel.recv_exit_status()

            # Print what stdout and stderr of the channel as soon as it appears
            # Not possible if stdout and stderr are redirected via '&> logfile.log'
            # for line in iter(stdout.readline, ""):
            #     print(f'[H]\'{thread_name}\': {line}', end="")

            elapsed_calc = time.time() - start_calc
            print(f'[T]\'{thread_name}\': Remote computation finished. Elapsed time: {elapsed_calc:.3f} seconds')

            # Set result status for each index to 'done', if a corresponding result file exists and is not empty
            # otherwise set result status to 'failed'
            print(f'[T]\'{thread_name}\': Updating grid status')
            for idx in param_idx:
                res_file = f'{grid_res_dir}/CGS_result_{grid_name}_idx_{idx}.csv'
                grid.at[idx, 'worker'] = thread_name
                try:
                    if os.path.getsize(res_file) > 0:
                        grid.at[idx, 'status'] = 'done'
                    else:
                        grid.at[idx, 'status'] = 'failed'
                except OSError as e:
                    grid.at[idx, 'status'] = 'failed'

        # End of while loop
        print(f'[T]\'{thread_name}\': No more parameter combinations available!')

    @staticmethod
    def __get_hardware_spec(client):
        stdin, stdout, stderr = client.exec_command("lscpu | grep 'Model name'")
        cpu = stdout.readline().split()
        cpu = ' '.join(map(str, cpu[2:]))
        print(f'CPU: {cpu}')

        stdin, stdout, stderr = client.exec_command("lscpu | grep 'CPU(s)'")
        num_cpu_cores = int(stdout.readline().split(":")[1])
        print(f'Cores: {num_cpu_cores}')

        stdin, stdout, stderr = client.exec_command("lscpu | grep 'min MHz'")
        cpu_min = float(stdout.readline().split(":")[1])
        print(f'CPU min: {cpu_min} MHz')

        stdin, stdout, stderr = client.exec_command("lscpu | grep 'max MHz'")
        cpu_max = float(stdout.readline().split(":")[1])
        print(f'CPU max: {cpu_max} MHz')

        stdin, stdout, stderr = client.exec_command("free -m | grep 'Mem'")
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
    def __ssh_connect(node, username, password=None):
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

    @staticmethod
    def __fetch_param_idx(param_grid, lock=None, num_params=1, set_status=True):
        """Fetch indices of the first num_params rows of param_grid that's status-column equals 'unsolved'

        Parameters
        ----------
        param_grid
            Linearized parameter grid of type pandas.DataFrame.
        lock
            A given lock makes sure, that all of the code inside the function is executed without switching between threads
        num_params
            Number of indices to fetch from param_grid. Is 1 by default.
        set_status
            If True, sets 'status' key of the fetched rows to 'pending', to exclude them from future calls.
            Can be used to check param_grid for fetchable or existent keys without changing their 'status' key.
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


def create_cgs_config(fp, circuit_template, param_map, dt, simulation_time, inputs,
                      outputs, sampling_step_size=None, **kwargs):
    """Creates a configfile.json containing a config_dict{} with input parameters as key-value pairs

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


def check_key_consistency(param_grid, param_map):
    grid_key_lst = list(param_grid.keys())
    map_key_lst = list(param_map.keys())
    return all((map_key in grid_key_lst for map_key in map_key_lst))


def gather_cgs_results(res_dir, num_header_params, filter_grid=None):
    header = list(range(num_header_params))
    files = glob.glob(res_dir + "/*.csv")

    if filter_grid:
        filter_grid = filter_grid.values.tolist()

    list_ = []
    for file_ in files:
        df = pd.read_csv(file_, index_col=0, header=header)
        if filter_grid:
            idx = list(df.columns.tolist()[0][:-1])
            idx = list(map(float, idx))
            if idx in filter_grid:
                list_.append(df)
        else:
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
