
# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural 
# network model_templates and simulations. See also:
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
"""Functions for performing parameter grid simulations with pyrates model_templates.
"""

# external imports
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from copy import deepcopy

# system imports
import os
import sys
import time as t
import glob
import getpass
import argparse
from pathlib import Path
from datetime import datetime
from threading import Thread, currentThread, RLock
import socket

# pyrates internal imports
from pyrates.frontend import CircuitTemplate
from pyrates.ir.circuit import CircuitIR

# meta infos
__author__ = "Christoph Salomon, Richard Gast"
__status__ = "development"


def grid_search(circuit_template: Union[CircuitTemplate, str], param_grid: Union[dict, pd.DataFrame], param_map: dict,
                step_size: float, simulation_time: float, inputs: dict, outputs: dict,
                sampling_step_size: Optional[float] = None, permute_grid: bool = False, init_kwargs: dict = None,
                clear: bool = True, **kwargs) -> tuple:
    """Function that runs multiple parametrizations of the same circuit in parallel and returns a combined output.

    Parameters
    ----------
    circuit_template
        Path to the circuit template.
    param_grid
        Key-value pairs for each circuit parameter that should be altered over different circuit parametrizations.
    param_map
        Key-value pairs that map the keys of param_grid to concrete circuit variables.
    step_size
        Simulation step-size in s.
    simulation_time
        Simulation time in s.
    inputs
        Inputs as provided to the `run` method of `:class:ComputeGraph`.
    outputs
        Outputs as provided to the `run` method of `:class:ComputeGraph`.
    sampling_step_size
        Sampling step-size as provided to the `run` method of `:class:ComputeGraph`.
    permute_grid
        If true, all combinations of the provided param_grid values will be realized. If false, the param_grid values
        will be traversed pairwise.
    clear
        If true, all files that have been created by PyRates to run the grid-search will be cleaned up afterwards.
    kwargs
        Additional keyword arguments passed to the `:class:ComputeGraph` initialization.


    Returns
    -------
    tuple
        Simulation results stored in a multi-index data frame, the mapping between the data frame column names and the
        parameter grid, the simulation time, and the memory consumption.

    """

    # argument pre-processing
    #########################

    if not init_kwargs:
        init_kwargs = {}
    vectorization = init_kwargs.pop('vectorization', True)
    if type(circuit_template) is str:
        circuit_template = CircuitTemplate.from_yaml(circuit_template)

    # linearize parameter grid if necessary
    if type(param_grid) is dict:
        param_grid = linearize_grid(param_grid, permute_grid)

    # create grid-structure of network
    ##################################

    # get parameter names and grid length
    param_keys = list(param_grid.keys())
    N = param_grid.shape[0]

    # assign parameter updates to each circuit, combine them to unconnected network and remember their parameters
    circuit = CircuitIR()
    circuit_names = []
    for idx in param_grid.index:
        new_params = {}
        for key in param_keys:
            new_params[key] = param_grid[key][idx]
        circuit_key = f'{circuit_template.label}_{idx}'
        circuit_tmp = adapt_circuit(deepcopy(circuit_template).apply(), new_params, param_map)
        circuit.add_circuit(circuit_key, circuit_tmp)
        circuit_names.append(circuit_key)
    param_grid.index = circuit_names

    # create backend graph
    net = circuit.compile(vectorization=vectorization, **init_kwargs)

    # adjust input of simulation to combined network
    for inp_key, inp in inputs.copy().items():
        inputs[f"all/{inp_key}"] = np.tile(inp, (1, N))
        inputs.pop(inp_key)

    # adjust output of simulation to combined network
    for out_key, out in outputs.items():
        outputs[out_key] = f"all/{out}"

    # simulate the circuits behavior
    results = net.run(simulation_time=simulation_time,
                      step_size=step_size,
                      sampling_step_size=sampling_step_size,
                      inputs=inputs,
                      outputs=outputs,
                      **kwargs)    # type: pd.DataFrame

    # clean up config files
    if clear:
        net.clear()

    # return results
    if 'profile' in kwargs:
        results, duration = results
        return results, param_grid, duration
    return results, param_grid


class ClusterCompute:
    def __init__(self, nodes: list, compute_dir=None, verbose: Optional[bool] = True):
        """Connect to nodes inside the computer network and create a compute directory with a unique compute ID

        Creates a compute directory for the `:class:ClusterCompute` instance in the specified path or as a default
        folder in the current working directory of the executing script.
        Creates a logfile in /ComputeDirectory/Logs and tees all future stdout and stderr of the executing script
        to this file
        Connects to all nodes via SSH and saves the corresponding paramiko client

        Parameters
        ----------
        nodes:
            List of names or IP addresses of working stations/servers in the local network
        compute_dir:
            Directory that will be used to store the logfiles.
            If none is provided, a default compute directory is created in the current working directory
        verbose:
            If False, all std output will still be copied to the log file but won't be shown in the terminal. Overwrites
            the verbosity of the run() method of all child classes.

        Returns
        -------

        """

        self.clients = []
        self.lock = RLock()

        #############################################################
        # Create main compute directory, subdirectories and logfile #
        #############################################################

        # Unique compute ID based on date and time
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
        if verbose:
            sys.stdout = StreamTee(sys.stdout, self.global_logfile)
            sys.stderr = StreamTee(sys.stderr, self.global_logfile)
        else:
            sys.stdout = open(self.global_logfile, 'a')
            sys.stderr = StreamTee(sys.stdout, self.global_logfile)

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
            try:
                client["paramiko_client"].close()
            except TypeError:
                pass

    def cluster_connect(self, nodes: list):
        """Create SSH connections to all nodes in the list, respectively

        Connect to all nodes in the given list via SSH, using the `ssh_connect` method of `:class:ClusterComoute`.
        Adds a dictionary for each node to the class internal 'clients' list
        Each dictionary contains:
            - ["paramiko_client"]: A paramiko client that can be used to execute commands on the node
            - ["node_name"]: The computer name of the connected node
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

    def spawn_thread(self, client, thread_kwargs, timeout):
        """

        Parameters
        ----------
        client
        thread_kwargs
        timeout

        Returns
        -------

        """
        t_ = Thread(
            name=client["node_name"],
            target=self.thread_master,
            args=(client, thread_kwargs, timeout)
        )
        t_.start()
        return t_

    def thread_master(self, client, thread_kwargs_: dict, timeout):
        """Function that is executed by every thread. Every instance of `thread_master` is bound to a different client

        The `thread_master` method of `:class:ClusterCompute` can be arbitrarily changed to fit the needs of the user.
        Commandline arguments on the remote node can be executed using client["paramiko_client"].exec_command("").

        Params
        ------
        client
            dict containing a `paramiko` client, the name of the connected node, a dict with hardware specifications and
            a path to a logfile.
        thread_kwargs
        timeout

        Returns
        -------

        """
        pass

    @staticmethod
    def get_hardware_spec(pm_client):
        """Print hardware specifications of the workstation tied to the paramiko client

        Prints CPU model, number of CPU cores, min CPU freq, max CPU freq and working memory of a paramiko client
        connected to a remote worker node

        Parameters
        ----------
        pm_client
            Paramiko client

        Returns
        -------
        dict


        """

        try:
            stdin, stdout, stderr = pm_client.exec_command("lscpu | grep 'Model name'")
            cpu = stdout.readline().split()
            cpu = ' '.join(map(str, cpu[2:]))
        except IndexError:
            cpu = "N/A"
        print(f'CPU: {cpu}')

        try:
            stdin, stdout, stderr = pm_client.exec_command("lscpu | grep 'CPU(s)'")
            num_cpu_cores = int(stdout.readline().split(":")[1])
        except IndexError:
            num_cpu_cores = "N/A"
        print(f'Cores: {num_cpu_cores}')

        try:
            stdin, stdout, stderr = pm_client.exec_command("lscpu | grep 'max MHz'")
            cpu_freq = float(stdout.readline().split(":")[1])
        except IndexError:
            cpu_freq = "N/A"
        print(f'CPU freq: {cpu_freq} MHz')

        try:
            stdin, stdout, stderr = pm_client.exec_command("free -m | grep 'Mem'")
            total_mem = int(stdout.readline().split()[1])
        except IndexError:
            total_mem = "N/A"
        print(f'Total memory: {total_mem} MByte')

        print("")

        hw = {
            'cpu': cpu,
            'num_cpu_cores': num_cpu_cores,
            'cpu_freq': cpu_freq,
            'total_mem': total_mem
        }
        return hw

    @staticmethod
    def ssh_connect(node, username, password=None, print_status=True):
        """Connect to a host via SSH and return a respective `paramiko.SSHClient`

        Parameters
        ----------
        node
            Name or IP-address of the host to connect to
        username
        password

        Returns
        -------
        paramiko.SSHClient
            Is None if connection fails. For detailed information of the thrown exception see Paramiko documentation

        """
        import paramiko
        client = paramiko.SSHClient()
        try:
            # Using kerberos authentication
            client.connect(node, username=username, gss_auth=True, gss_kex=True)
            if print_status:
                print(f'\'{node}\': Connection established!')
            return client
        except paramiko.ssh_exception.NoValidConnectionsError as err:
            if print_status:
                print(f'\'{node}\': ', err)
            return None
        except paramiko.ssh_exception.AuthenticationException as err:
            if print_status:
                print(f'\'{node}\': ', err)
            return None
        except paramiko.ssh_exception.SSHException as err:
            if print_status:
                print(f'\'{node}\': ', err)
            return None
        except IOError as err:
            if print_status:
                print(f'\'{node}\': ', err)
            return None


class ClusterGridSearch(ClusterCompute):
    def __init__(self, nodes, compute_dir=None, verbose: Optional[bool] = True):
        """Connect to nodes inside the computer network and create a compute directory with a unique compute ID

        The use of `:class:ClusterGridSearch` requires the utilized computer network to feature two major properties:
        1. A GSSAPI-based authentication method (e.g. Kerberos).
           Since no secure password check is implemented in `:class:ClusterGridSearch`, it uses `paramiko's` gssapi
           module to authenticate users when an SSH connection is established with a worker.
           This has to be supported by the utilized computer network
        2. A server-based file system.
           After an SHH-connection has been established with a worker, all files that reside on the master have to be
           accessible by the worker using the same absolute file paths.
           A transfer of the necessary files from the master to the worker is generally possible using paramiko's SFTP
           module, but has not been implemented in `:class:ClusterGridSearch`.


        The compute directory contains the following sub folders:
        /Builds: Contains the build directories that are created by the `grid_search` function for each worker
        /Config: Contains configuration files in .json format that yield PyRates simulation parameters for each
                 simulation that has been performed by the `:class:ClusterGridSearch` instance.
        /Grids: Contains all parameter grids as .csv files that have been computed by the current CGS instance and their
                respective sub grids that were transferred to the workers as .h5 files.
        /Logs:  Contains the global log file and the local log files of each worker for each call to the `run` method of
                the current `:class:ClusterGridSearch` instance.
                The local log files contain the stdout and the stderr of each worker, respectively
                The global log files contains the stdout and the stderr of the master
        /Results: Contains all results in .h5 format that have been produced by the `:class:ClusterGridSearch` instance.
                  During a computation it also contains the intermediate local result files of the workers.
                  These files are deleted once the global result file created.

        Stdout and stderr of the worker will be teed to a global log file that resides in the compute directory.
        If the verbosity argument is set to False, all future stdout and stderr of the CGS instance will no longer be
        printed to the terminal, independent of the verbose argument of the call to the `run` method, but still be
        written to the global log file.

        Parameters
        ----------
        nodes
            List of names or IP addresses of working stations/servers in the local network
        compute_dir
            Directory that will be used to store the logfiles.
            If none is provided, a default compute directory is created in the current working directory
        verbose
            If False, all std output will still be copied to the log file but won't be shown in the terminal

        Returns
        -------

        """

        super().__init__(nodes, compute_dir, verbose)

        self.chunk_idx = 0
        self.res_file_collection = {}

        # Adding additional subfolders to the compute directory
        # Grid directory
        self.grid_dir = f'{self.compute_dir}/Grids'
        os.makedirs(self.grid_dir, exist_ok=True)

        # Config directory
        self.config_dir = f'{self.compute_dir}/Config'
        os.makedirs(self.config_dir, exist_ok=True)

        # Result directory
        self.res_dir = f'{self.compute_dir}/Results'
        os.makedirs(self.res_dir, exist_ok=True)

        # Build directory for pyrates backend
        self.build_dir = f'{self.compute_dir}/Builds'
        os.makedirs(self.build_dir, exist_ok=True)

    def run(self, circuit_template: str, param_grid: Union[dict, pd.DataFrame], param_map: dict, dt: float,
            simulation_time: float, chunk_size: (int, list),  inputs: dict, outputs: dict,
            worker_env: Optional[Union[str, list]] = sys.executable, worker_file: Optional[str] = os.path.abspath(__file__),
            sampling_step_size: Optional[float] = None, result_kwargs: Optional[dict] = {},
            worker_kwargs: Optional[dict] = {}, gs_kwargs: Optional[dict] = {},
            add_template_info: Optional[bool] = False, permute_grid: Optional[bool] = False,
            verbose: Optional[bool] = True, **kwargs) -> str:
        """Starts multiple threads on the master and distributes a parameter grid among nodes in a compute cluster that
        concurrently run a grid_search computation on their provided parameter sub grids.

        Intermediate result files will be concatenated to a single result file after all computations have finished.
        The global result file is a hierarchical data format (hdf5) file that contains the following groups and data
        sets.
        All groups that are marked with (df) can be loaded into a pandas.DataFrame, using pandas.read_hdf():

        /Config
            /circuit_template
            /config_file
            /step_size
            /sampling_step_size
            /simulation_time
            /inputs (if provided)
        /ParameterGrid
            /Grid_df (df)
            /Keys
                /Parameter_1
                / ...
                /Parameter_p
        /Results
            /result_map (df)
            /results (df)
        /AdditionalData
            /result_kwargs_key_0
            /...
            /result_kwargs_key_k
        /TemplateInfo (if add_template_info is True)

        e.g. to access the stored simulation you can
        and performs Run multiple instances of grid_search simultaneously on different workstations in the compute cluster

        `:class:ClusterGridSearch` requires all necessary data (worker script, worker environment, compute directory)
        to be stored on a server. After a connection is established, all data on the remote workers has to be accessible
        in the same way as it would be on the master worker (same root directory, same filepaths, etc.)
        Paths inside the run function have to be adjusted respectively.

        FTP implementation is planned in future versions for direct file transfer between the master and the workers.


        Parameters
        ----------
        circuit_template
            Path to the circuit template file.
        param_grid
            Key-value pairs for each circuit parameter that should be altered over different circuit parametrizations.
        param_map
            Key-value pairs that map the keys of param_grid to circuit variables.
        dt
            Simulation step-size in s.
        simulation_time
            Simulation time in s.
        inputs
            inputs as provided to the `run` method of `:class:ComputeGraph`.
        outputs
            Outputs as provided to the `run` method of `:class:ComputeGraph`.
        sampling_step_size
            Sampling step-size as provided to the `run` method of `:class:ComputeGraph`.
        permute_grid
            If true, all combinations of the provided param_grid values will be realized. If false, the param_grid
            values will be traversed pairwise.
        chunk_size
            int or list that defines how many parametrizations are computed at once by each worker.
            If int, every worker fetches the specified amount of parametrizations. Worker repeatedly fetch parameter
            chunks until the whole param_grid has been computed.
            If list, ever worker fetches an individual amount of parametrizations during each computation.
            The order of chunk sizes in the chunk_size list is mapped to the order of nodes in the node list.
            Their lengths have to match.
        worker_env
            Path to python executable inside an environment that will be used to execute the worker file.
            Can be used if worker file with customized post processing should be executed in a different environment
            than the executing script.
        worker_file
            Path to customized worker file.
        result_kwargs
            Key-value pairs that will be added to the 'AdditionalData' group inside the result file.
        worker_kwargs
            Key-value pairs that will be added to the config file.
        gs_kwargs
            Key-Value pairs that will be passed to the grid_search() call of the workers
        add_template_info
            If True, all operator templates and its variables are copied from the circuit_template yaml file to the
            'TemplateInfo' group inside the result file.
        verbose
            If False, stdout and stderr will be copied to the log file but not be printed to the terminal

        Returns
        -------
        String containing location of the result file
        """

        import h5py

        if verbose:
            sys.stdout = StreamTee(sys.stdout, self.global_logfile)
            sys.stderr = StreamTee(sys.stderr, self.global_logfile)
        else:
            sys.stdout = open(self.global_logfile, 'a')
            sys.stderr = StreamTee(sys.stderr, self.global_logfile)

        t_total = t.time()

        print("")

        # Prepare parameter grid
        ########################
        print("***PREPARING PARAMETER GRID***")
        t0 = t.time()

        # Create DataFrame from param dictionary
        if isinstance(param_grid, dict):
            param_grid = linearize_grid(param_grid, permute=permute_grid)

        # Create default parameter grid csv-file
        grid_idx = 0
        while os.path.exists(f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'):
            grid_idx += 1
        grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'
        grid_name = Path(grid_file).stem
        param_grid.to_csv(grid_file, index=True)

        # Add desired chunk size to each node
        #####################################
        for i, client in enumerate(self.clients):
            if isinstance(chunk_size, list):
                client['chunk_size'] = chunk_size[i]
            else:
                client['chunk_size'] = chunk_size

            if isinstance(worker_env, list):
                client['worker_env'] = worker_env[i]
            else:
                client['worker_env'] = worker_env

        # Assign chunk indices to parameter grid
        working_grid = param_grid.copy()
        working_grid["status"] = 'unsolved'
        working_grid["chunk_idx"] = -1
        working_grid["err_count"] = 0

        print(f'Done! Elapsed time: {t.time() - t0:.3f} seconds')

        print("")

        # Check key consistency
        #######################
        print("***CHECKING PARAMETER GRID AND MAP FOR CONSISTENCY***")
        t0 = t.time()
        if not self.check_key_consistency(param_grid, param_map):
            print("WARNING: Not all parameter grid keys found in parameter map (see above). This needs to be handled "
                  "accordingly by the user-specified worker template. Else, the network construction will fail with "
                  "a key error.")
        print(f'Done! Elapsed time: {t.time() - t0:.3f} seconds')

        print("")

        # Create config file
        #####################
        print("***CREATING CONFIG FILE***")
        t0 = t.time()
        config_idx = 0
        while os.path.exists(f'{self.config_dir}/DefaultConfig_{config_idx}.json'):
            config_idx += 1
        config_file = f'{self.config_dir}/DefaultConfig_{config_idx}.yaml'
        self.create_cgs_config(fp=config_file,
                               circuit_template=circuit_template,
                               param_map=param_map,
                               dt=dt,
                               simulation_time=simulation_time,
                               inputs=inputs,
                               outputs=outputs,
                               sampling_step_size=sampling_step_size,
                               gs_kwargs=gs_kwargs,
                               worker_kwargs=worker_kwargs)
        print(f'Done! Elapsed time: {t.time() - t0:.3f} seconds')

        print("")

        # Create global result file
        ############################
        print("***CREATING GLOBAL RESULT FILE***")
        t0 = t.time()
        # Create result directory and result file for current parameter grid
        grid_res_dir = f'{self.res_dir}/{grid_name}'
        os.makedirs(grid_res_dir, exist_ok=True)
        global_res_file = f'{grid_res_dir}/CGS_result_{grid_name}.h5'

        # Write grid and config information to global result file
        with h5py.File(global_res_file, 'a') as file:
            for key, value in param_grid.items():
                file.create_dataset(f'/ParameterGrid/Keys/{key}', data=value)
            for key, value in result_kwargs.items():
                file.create_dataset(f'/AdditionalData/{key}', data=value)
            file.create_dataset(f'/Config/config_file', data=config_file)
            file.create_dataset(f'/Config/circuit_template', data=circuit_template)
            file.create_dataset(f'/Config/simulation_time', data=simulation_time)
            file.create_dataset(f'/Config/step_size', data=dt)
            file.create_dataset(f'/Config/sampling_step_size', data=sampling_step_size)
            if add_template_info:
                template_file = f'{circuit_template.rsplit("/", 1)[0]}.yaml'
                self.add_template_information(template_file, file)

        param_grid.to_hdf(global_res_file, key='/ParameterGrid/Grid_df')
        print(f'Done. Elapsed time: {t.time() - t0:.3f} seconds')

        print("")

        # Create keyword dictionary for threads
        #######################################
        thread_kwargs = {
            "worker_env": worker_env,
            "worker_file": worker_file,
            "working_grid": working_grid,
            "grid_name": grid_name,
            "grid_res_dir": grid_res_dir,
            "config_file": config_file,
            "global_res_file": global_res_file
        }

        # Start cluster computation
        ###########################
        print("***STARTING CLUSTER COMPUTATION***")
        # Spawn threads to control each node connection and start computation
        timeout = worker_kwargs['time_lim'] if 'time_lim' in worker_kwargs else None
        threads = [self.spawn_thread(client, thread_kwargs, timeout=timeout) for client in self.clients]
        # Wait for all threads to finish
        for t_ in threads:
            t_.join()

        print("")
        print(f'Cluster computation finished. Elapsed time: {t.time() - t_total:.3f} seconds')

        # Check parameter combinations that failed multiple times to compute
        remaining_params = working_grid.loc[working_grid["status"] == "failed"].index
        if len(remaining_params) > 0:
            print("WARNING: The following parameter indices could not be computed: ")
            print(remaining_params)
        print("")

        # Write local results to global result file
        ###########################################
        print(f'***WRITING RESULTS TO GLOBAL RESULT FILE***')
        t0 = t.time()

        # Get sorted list of temporary result files to iterate through
        temp_res_files = glob.glob(f'{grid_res_dir}/*_temp*')
        temp_res_files.sort()

        # Read number of different circuit outputs and prepare lists to concatenate results
        res_dict = {}
        try:
            with pd.HDFStore(temp_res_files[0], "r") as store:
                for key in store.keys():
                    res_dict[key] = []
        except IndexError:
            print("No result file created. Check local log files for worker script errors.")
            return ""

        # Concatenate results from each temporary result file
        for file in temp_res_files:
            with pd.HDFStore(file, "r") as store:
                for idx, key in enumerate(store.keys()):
                    res_dict[key].append(store[key])

        # Create DataFrame for each output variable and write to global result file
        with pd.HDFStore(global_res_file, "a") as store:
            for key, value in res_dict.items():
                if key != '/result_map' and len(value) > 0:
                    df = pd.concat(value, axis=kwargs.pop('result_concat_axis', 1))
                    store.put(key=f'/Results{key}', value=df)
            result_map = pd.concat(res_dict['/result_map'], axis=0)
            store.put(key=f'/Results/result_map', value=result_map)

        # Delete temporary local result files
        for file in temp_res_files:
            os.remove(file)

        print(f'Elapsed time: {t.time()-t0:.3f} seconds')
        print(f'Find results in: {grid_res_dir}')
        print("")

        return global_res_file

    def thread_master(self, client: dict, thread_kwargs: dict, timeout: float):
        """Function that is executed by each thread to schedule computations on the respective worker

        Parameters
        ----------
        client
            Dictionary containing all information about the remote worker that is tied to the current thread
        thread_kwargs
            Dictionary containing all kwargs that are passed to the thread function
        timeout

        Returns
        -------

        """

        import paramiko

        # This lock ensures that parameter chunks are fetched by workers in the same order as they are defined
        # in the node list
        self.lock.acquire()

        thread_name = currentThread().getName()
        connection_lost = False
        timed_out = False
        connection_lost_counter = 0

        # Get client information
        pm_client = client["paramiko_client"]
        logfile = client["logfile"]
        chunk_size = client["chunk_size"]
        worker_env = client["worker_env"]

        # Get keyword arguments
        config_file = thread_kwargs["config_file"]
        working_grid = thread_kwargs["working_grid"]
        grid_name = thread_kwargs["grid_name"]
        grid_res_dir = thread_kwargs["grid_res_dir"]
        worker_file = thread_kwargs["worker_file"]

        # Prepare worker command
        command = f'{worker_env} {worker_file}'

        # Create folder to save local subgrids to
        subgrid_dir = f'{self.grid_dir}/Subgrids/{grid_name}/{thread_name}'
        os.makedirs(subgrid_dir, exist_ok=True)
        subgrid_idx = 0

        # Create build_dir for grid_search numpy backend
        local_build_dir = f'{self.build_dir}/{thread_name}'
        os.makedirs(local_build_dir, exist_ok=True)

        # Start scheduler
        #################
        while True:
            # Disable thread switching
            self.lock.acquire()

            # Fetch grid chunks
            ####################
            # Get all parameters that haven't been successfully computed yet
            remaining_params = working_grid.loc[working_grid["status"] == "unsolved"]
            if remaining_params.empty:
                # check for pending parameters
                pending_params = working_grid.loc[working_grid["status"] == "pending"]
                if pending_params.empty:
                    print(f'[T]\'{thread_name}\': No more parameter combinations available!')
                    # Enable thread switching
                    self.lock.release()
                    # Release the initial lock in the first iteration
                    try:
                        self.lock.release()
                    except RuntimeError:
                        pass
                    break
                else:
                    # Enable thread switching
                    self.lock.release()
                    # Release the initial lock in the first iteration
                    try:
                        self.lock.release()
                    except RuntimeError:
                        pass
                    t.sleep(10.0)
                    continue
            else:
                # Find chunk index of first value in remaining params and fetch all parameter combinations with the
                # same chunk index
                param_idx, chunk_idx = self._fetch_index(remaining_params, chunk_size, working_grid)
                working_grid.at[param_idx, "status"] = "pending"

                print(f'[T]\'{thread_name}\': Fetching {len(param_idx)} indices: ', end="")
                print(f'[{param_idx[0]}] - [{param_idx[-1]}]')

                # Create parameter sub-grid
                ###########################
                subgrid_fp = f'{subgrid_dir}/{thread_name}_Subgrid_{subgrid_idx}.h5'
                subgrid_df = working_grid.loc[param_idx, :]
                subgrid_df.to_hdf(subgrid_fp, key="subgrid")
                subgrid_idx += 1

                # Temporary result file for current subgrid
                ###########################################
                local_res_file = f'{grid_res_dir}/CGS_result_{grid_name}_' \
                    f'chunk_{chunk_idx:02}_idx_{param_idx[0]}-{param_idx[-1]}_temp.h5'
                self.res_file_collection[chunk_idx] = local_res_file

                # Execute worker script on the remote host
                ##########################################
                t0 = t.time()

                try:
                    print(f'[T]\'{thread_name}\': Starting remote computation...')

                    channel = pm_client.get_transport().open_session()
                    channel.settimeout(timeout)

                    # Execute the given command
                    channel.get_pty()
                    channel.exec_command(command +
                                         f' --config_file={config_file}'
                                         f' --subgrid={subgrid_fp}'
                                         f' --local_res_file={local_res_file}'
                                         f' --build_dir={local_build_dir}'
                                         f' &>{logfile}',  # redirect and append stdout
                                         )

                    # stdin, stdout, stderr = pm_client.exec_command(command +
                    #                                                f' --config_file={config_file}'
                    #                                                f' --subgrid={subgrid_fp}'
                    #                                                f' --local_res_file={local_res_file}'
                    #                                                f' --build_dir={local_build_dir}'
                    #                                                f' &>> {logfile}',  # redirect and append stdout
                    #                                                                    # and stderr to logfile
                    #                                                timeout=timeout,        # timeout in seconds
                    #                                                get_pty=True)       # execute in pseudo terminal

                except (socket.timeout, paramiko.ssh_exception.SSHException,  ConnectionError, EOFError) as e:
                    # SSH connection has been lost or process ran into timeout
                    print(f'[T]\'{thread_name}\': ERROR: {e}')
                    working_grid.at[param_idx, "status"] = "unsolved"
                    connection_lost = True
                    timed_out = e == socket.timeout

            # Enable thread switching
            self.lock.release()

            # Release the second lock if its still acquired from the very beginning
            try:
                self.lock.release()
            except RuntimeError:
                pass

            # Try to reconnect to host if connection has been lost
            ######################################################
            if connection_lost and connection_lost_counter < 2 and not timed_out:
                # Attempt to reconnect while there are still parameter chunks to fetch
                while True:
                    if working_grid.loc[working_grid["status"] == "unsolved"].empty:
                        # Stop thread execution if no more parameters are available
                        return
                    else:
                        print('Attempting to reconnect')
                        pm_client = self.ssh_connect(thread_name, username=getpass.getuser(), print_status=False)
                        if pm_client:
                            print(f'[T]\'{thread_name}\': Reconnected!')
                            connection_lost = False
                            # Escape reconnection loop
                            break
                        t.sleep(30)
                # Jump to the beginning of scheduler loop if reconnection was successful
                continue

            # Wait for remote computation exit status
            # (independent of success or failure)
            #########################################

            nbytes = 1024
            t1 = t.time()
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    data = channel.recv(nbytes)
                    while data:
                        data = channel.recv(nbytes)
                if channel.recv_stderr_ready():
                    error_buff = channel.recv_stderr(nbytes)
                    while error_buff:
                        error_buff = channel.recv_stderr(nbytes)
                t2 = t.time()
                if t2-t1 > timeout:
                    channel.close()
                    timed_out = True
                    break
                t.sleep(1.0)
            if timed_out:
                exit_status = -1
            else:
                channel.close()
                exit_status = channel.recv_exit_status()

            # Update grid status
            ####################
            with self.lock:
                # Remote machine executed worker script successfully
                if exit_status == 0:
                    print(f'[T]\'{thread_name}\': Remote computation finished. Elapsed time: {t.time()-t0:.3f} seconds')
                    print(f'[T]\'{thread_name}\': Updating grid status')
                    try:
                        # Check if data has been written to result file
                        with pd.HDFStore(local_res_file, "r") as store:
                            # Check if temporary result file (hdf5) contains keys
                            if len(store.keys()) == 0:
                                raise KeyError
                            # Check if each key contains data
                            for key in store.keys():
                                if len(store[key].index) == 0:
                                    raise KeyError
                        working_grid.at[param_idx, 'status'] = 'done'
                    except (KeyError, FileNotFoundError):
                        for row in param_idx:
                            if working_grid.at[row, "err_count"] < 2:
                                working_grid.at[row, "status"] = "unsolved"
                                working_grid.at[row, "err_count"] += 1
                            else:
                                working_grid.at[row, "status"] = "failed"

                # Remote execution was interrupted or remote process ran into timeout
                else:
                    print(f'[T]\'{thread_name}\': Remote computation failed with exit status {exit_status}')
                    connection_lost_counter += 1
                    for row in param_idx:
                        if working_grid.at[row, "err_count"] < 4:
                            working_grid.at[row, "status"] = "unsolved"
                            working_grid.at[row, "err_count"] += 1
                        else:
                            working_grid.at[row, "status"] = "failed"
                    if connection_lost_counter >= 1 or timed_out:
                        print('Excluding worker from pool')
                        return

            # Lock released, thread switching enabled
        # End of scheduler loop
    # End of Thread master

    def _fetch_index(self, remaining_params, worker_chunk_size, working_grid):
        param_idx = []
        chunk_count = 0
        chunk_idx = self.chunk_idx
        for i, row in remaining_params.iterrows():
            if chunk_count == worker_chunk_size:
                self.chunk_idx += 1
                return pd.Index(param_idx), chunk_idx
            elif row['chunk_idx'] == -1:
                # Chunk parameter grid
                working_grid.at[i, 'chunk_idx'] = self.chunk_idx
                param_idx.append(i)
                chunk_count += 1
            else:
                pass
        # Whole parameter grid is chunked
        if len(param_idx) == 0:
            # get all parameters with the same valid chunk index
            chunk_idx = remaining_params.iloc[0]["chunk_idx"]
            param_idx = remaining_params.loc[remaining_params["chunk_idx"] == chunk_idx].index

        return pd.Index(param_idx), chunk_idx

    #################################
    # CGS internal helper functions #
    #################################
    @staticmethod
    def chunk_grid(grid: pd.DataFrame, chunk_size: int):
        """Create a pd.DataFrame from the params dict and assign a chunk index to each parameter combination

        Parameters
        ----------
        grid
            pd.DataFrame with all parameter combinations that can be traversed linearly
        chunk_size
            Number of parameter combinations to be computed simultaneously on one worker

        Returns
        -------
            pd.DataFrame

        """
        # Add status columns to grid
        chunked_grid = grid.copy()
        chunked_grid["status"] = 'unsolved'
        chunked_grid["chunk_idx"] = -1
        chunked_grid["err_count"] = 0

        # Assign a chunk index to each parameter combination
        chunk_count = 0
        while True:
            param_idx = chunked_grid.loc[chunked_grid["status"] == "unsolved"].index[:chunk_size]
            if param_idx.empty:
                break
            else:
                chunked_grid.at[param_idx, 'chunk_idx'] = chunk_count
                chunked_grid.at[param_idx, 'status'] = "chunked"
                chunk_count += 1
        chunked_grid["status"] = 'unsolved'

        return chunked_grid

    @staticmethod
    def check_key_consistency(param_grid: Union[dict, pd.DataFrame], param_map: dict):
        """Check if keys in param_grid and param_map match

        Parameters
        ----------
        param_grid
        param_map

        Returns
        -------
        bool

        """
        grid_key_lst = list(param_grid.keys())
        map_key_lst = list(param_map.keys())
        # Not all keys of parameter map can be found in parameter grid
        if not all((grid_key in map_key_lst for grid_key in grid_key_lst)):
            print("Parameter map keys:")
            print(*list(param_map.keys()))
            print("Parameter grid keys:")
            print(*list(param_grid.keys()))
            return False
        # Parameter grid and parameter map match
        else:
            print("All parameter grid keys found in parameter map")
            return True

    @staticmethod
    def create_cgs_config(fp: str, circuit_template: str, param_map: dict, dt: float, simulation_time: float,
                          sampling_step_size: Optional[float] = None, inputs: Optional[dict] = {},
                          outputs: Optional[dict] = {}, gs_kwargs: Optional[dict] = {}, 
                          worker_kwargs: Optional[dict] = {}):
        """Creates a configfile.json containing a dictionary with all input parameters as key-value pairs

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
        gs_kwargs
        worker_kwargs

        Returns
        -------

        """

        import yaml

        if sampling_step_size is None:
            sampling_step_size = dt

        config_dict = {
            "circuit_template": circuit_template,
            "param_map": param_map,
            "step_size": dt,
            "simulation_time": simulation_time,
            "inputs": inputs,
            "outputs": outputs,
            "sampling_step_size": sampling_step_size,
            "gs_kwargs": gs_kwargs,
            "worker_kwargs": worker_kwargs
        }

        with open(fp, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def add_template_information(yaml_fp, hdf5_file):
        """Add opearator information of the circuit template to the global result file"""

        import yaml

        with open(yaml_fp, 'r') as stream:
            try:
                yaml_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        for operator_key, operator_value in yaml_dict.items():
            if "Op" in operator_key:
                for temp_key, temp_value in operator_value.items():
                    if isinstance(temp_value, str):
                        hdf5_file.create_dataset(f'/TemplateInfo/{operator_key}/{temp_key}', data=temp_value)
                    if isinstance(temp_value, list):
                        for idx, eq in enumerate(temp_value):
                            hdf5_file.create_dataset(f'/TemplateInfo/{operator_key}/{temp_key}/eq_{idx}', data=eq)
                    elif isinstance(temp_value, dict):
                        for key, value in temp_value.items():
                            try:
                                hdf5_file.create_dataset(f'/TemplateInfo/{operator_key}/{temp_key}/{key}', data=value["default"])
                            except:
                                hdf5_file.create_dataset(f'/TemplateInfo/{operator_key}/{temp_key}/{key}', data=value)

#####################
# Utility functions #
#####################


def linearize_grid(grid: dict, permute: bool = False) -> pd.DataFrame:
    """Turns the grid into a grid that can be traversed linearly, i.e. pairwise.

    Parameters
    ----------
    grid
        Parameter grid.
    permute
        If true, all combinations of the parameter values in grid will be created.

    Returns
    -------
    pd.DataFrame
        Resulting linear grid in form of a data frame.

    """

    arg_lengths = [len(arg) for arg in grid.values()]

    if len(list(set(arg_lengths))) == 1 and not permute:
        return pd.DataFrame(grid)
    elif permute:
        vals, keys = [], []
        for key, val in grid.items():
            vals.append(val)
            keys.append(key)
        new_grid = np.stack(np.meshgrid(*tuple(vals)), -1).reshape(-1, len(grid))
        return pd.DataFrame(new_grid, columns=keys)
    else:
        raise ValueError('Wrong number of parameter combinations. If `permute` is False, all parameter vectors in grid '
                         'must have the same number of elements.')


def adapt_circuit(circuit: CircuitIR, params: dict, param_map: dict) -> CircuitIR:
    """Changes the parametrization of a circuit.

    Parameters
    ----------
    circuit
        Circuit instance.
    params
        Key-value pairs of the parameters that should be changed.
    param_map
        Map between the keys in params and the circuit variables.

    Returns
    -------
    CircuitIR
        Updated circuit instance.

    """

    for key in params.keys():

        val = params[key]

        for var in param_map[key]['vars']:

            # change variable values on nodes
            nodes = param_map[key]['nodes'] if 'nodes' in param_map[key] else []
            for node in nodes:
                circuit[node].values = deepcopy(circuit[node].values)
                if "/" in var:
                    op, var_name = var.split("/")
                    if op in circuit[node]:
                        circuit[node].values[op][var_name] = float(val)
                else:
                    for op, _ in circuit[node]:
                        try:
                            circuit[node].values[op][var] = float(val)
                        except KeyError:
                            print(f'WARNING: Variable {var} has not been found on node {node}.')

            # change variable values on edges
            edges = param_map[key]['edges'] if 'edges' in param_map[key] else []
            if edges and len(edges[0]) < 3:
                for source, target in edges:
                    if var in circuit.edges[source, target, 0]:
                        circuit.edges[source, target, 0][var] = float(val)
            else:
                for source, target, edge in edges:
                    if var in circuit.edges[source, target, edge]:
                        circuit.edges[source, target, edge][var] = float(val)

    return circuit


class StreamTee(object):
    """Tee all stdout to a specified logfile"""
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


##################
# Cluster worker #
##################


class ClusterWorkerTemplate(object):
    """Contains an interface for its methods to be called on a cluster worker by a remote master.

    """
    def __init__(self):
        self.FLAGS = argparse.Namespace
        self.results = pd.DataFrame
        self.result_map = pd.DataFrame
        self.processed_results = pd.DataFrame

    def worker_init(self, config_file="", subgrid="", result_file="", build_dir=os.getcwd()):
        """Interface to receive input when run from the console

        Returns
        -------

        """
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--config_file",
            type=str,
            default=config_file,
            help="File to load grid_search configuration parameters from"
        )

        parser.add_argument(
            "--subgrid",
            type=str,
            default=subgrid,
            help="File to load parameter grid from"
        )

        parser.add_argument(
            "--local_res_file",
            type=str,
            default=result_file,
            help="File to save results to"
        )

        parser.add_argument(
            "--build_dir",
            type=str,
            default=build_dir,
            help="Custom PyRates build directory"
        )

        self.FLAGS = parser.parse_args()
        self.worker_exec(sys.argv)

    def worker_exec(self, _):
        """Reads configurations and sub grid from the respective files and executes a grid_search() call with the
           provided parameters.

        Parameters
        ----------
        _

        Returns
        -------

        """
        import warnings
        # external imports
        from numba import config
        import yaml

        # tf.config.set_soft_device_placement(True)

        config.THREADING_LAYER = 'omp'

        # Disable general warnings
        warnings.filterwarnings("ignore")

        # disable TF-gpu warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        t_total = t.time()

        # Load command line arguments and create logfile
        ################################################
        print("")
        print("***LOADING COMMAND LINE ARGUMENTS***")
        t0 = t.time()

        config_file = self.FLAGS.config_file
        subgrid = self.FLAGS.subgrid
        local_res_file = self.FLAGS.local_res_file
        build_dir = self.FLAGS.build_dir

        print(f'Elapsed time: {t.time() - t0:.3f} seconds')

        # Load global config file
        #########################
        print("")
        print("***LOADING GLOBAL CONFIG FILE***")
        t0 = t.time()

        with open(config_file, "r") as g_conf:
            global_config_dict = yaml.load(stream=g_conf, Loader=yaml.UnsafeLoader)

        circuit_template = global_config_dict['circuit_template']
        param_map = global_config_dict['param_map']
        dt = global_config_dict['step_size']
        simulation_time = global_config_dict['simulation_time']
        sampling_step_size = global_config_dict['sampling_step_size']

        # Optional parameters
        #####################

        try:
            inputs = global_config_dict['inputs']
        except KeyError:
            inputs = {}

        try:
            outputs = global_config_dict['outputs']
        except KeyError:
            outputs = {}

        try:
            worker_kwargs = global_config_dict['worker_kwargs']
        except KeyError:
            worker_kwargs = {}

        try:
            gs_kwargs = global_config_dict['gs_kwargs']
        except KeyError:
            gs_kwargs = {}

        if 'init_kwargs' not in gs_kwargs:
            gs_kwargs['init_kwargs'] = {}
        if build_dir:
            gs_kwargs['init_kwargs']['build_dir'] = build_dir

        print(f'Elapsed time: {t.time() - t0:.3f} seconds')

        # restrict resources of worker process
        resource_kwargs = {}
        for key in ['cpu_lim', 'memory_lim', 'nproc_lim', 'time_lim']:
            if key in worker_kwargs:
                resource_kwargs[key] = worker_kwargs.pop(key)
        self._limit_resources(**resource_kwargs)

        # LOAD PARAMETER GRID
        #####################
        print("")
        print("***PREPARING PARAMETER GRID***")
        t0 = t.time()

        # Load subgrid into DataFrame
        param_grid = pd.read_hdf(subgrid, key="subgrid")

        # Drop all columns that don't contain a parameter map value (e.g. status, chunk_idx, err_count) since
        # grid_search() can't handle additional columns
        param_grid = param_grid[[key for key in param_map.keys() if key in param_grid]]
        worker_kwargs.update({'param_grid': param_grid})
        print(f'Elapsed time: {t.time() - t0:.3f} seconds')

        # COMPUTE PARAMETER GRID
        ########################
        print("")
        print("***COMPUTING PARAMETER GRID***")
        t_ = self.worker_gs(circuit_template=circuit_template,
                            param_grid=param_grid.copy(),
                            param_map=param_map,
                            simulation_time=simulation_time,
                            step_size=dt,
                            sampling_step_size=sampling_step_size,
                            permute_grid=False,
                            inputs=inputs,
                            outputs=outputs.copy(),
                            profile=True,
                            **gs_kwargs)

        print(f'Total parameter grid computation time: {t_:.3f} seconds')

        # Post process results and write data to local result file
        ##########################################################
        print("")
        print("***POSTPROCESSING AND CREATING RESULT FILES***")
        t0 = t.time()

        self.worker_postprocessing(**worker_kwargs)

        with pd.HDFStore(local_res_file, "w") as store:
            store.put(key='results', value=self.processed_results)
            store.put(key='result_map', value=self.result_map)

        print(f'Result files created. Elapsed time: {t.time() - t0:.3f} seconds')
        print("")
        print(f'Total elapsed time: {t.time() - t_total:.3f} seconds')

    def worker_gs(self, *args, **kwargs):
        """
        Function that contains the grid_search call. Can be customized if necessary, as long as the `results` and
        `result_map` attributes of the worker are updated during the call.

        Returns
        -------
        float
            simulation time
        """
        self.results, self.result_map, t_ = grid_search(*args, **kwargs)
        return t_

    def worker_postprocessing(self, **worker_kwargs):
        """
        Post processing that is applied by each cluster worker on its computed model output.

        The ClusterWorkerTemplate class contains three DataFrames that can be used to add customized post processing:

        To apply the customized post processing during a ClusterGridSearch call, a customized worker file has to
        be created that is passed to the CGS.run() call.
        For that purpose, execute the following steps:
        1. Create a new python script.
        2. Derive a child class from the ClusterWorkerTemplate class and customize its worker_postprocessing() method.
           Use self.results and self.result_map to access simulation data.
           Safe the post processed data to self.processed_results.

           self.results - DataFrame that contains the worker's grid_search() output for the provided parameter sub grid
           self.result_map - DataFrame that contains the computed parametrizations and their respective identification
                             key which is used in the column names of the results DataFrame
           self.processed_results - DataFrame that will be send back to the master.
                                    Has the same column structure as self.results, yet the index is unset.
                                    The processed_results of all workers will be concatenated to a the final result

        3. Ensure that if the worker script is executed, an instance of your customized worker class is invoked and that
           its 'worker_init()' method is called.

        When implementing customized post processing, use the following code to iterate over all computed results.
        Make sure to write all final data to self.processed_results:

        for idx, data in self.results.iteritems():

            # add processing of 'data' here

            self.processed_results[:, idx] = data

        An example worker template that estimates the power spectral density of each model parametrization results is
        presented below:

            custom_worker.py

            from pyrates.utility.grid_search import ClusterWorkerTemplate
            from scipy.signal import welch


            class MyWorker(ClusterWorkerTemplate):
                def worker_postprocessing(self):
                    for idx, data in self.results.iteritems():
                        t = self.results.index.to_list()
                        step_size = t[1] - t[0]
                        f, p = welch(data.to_numpy(), fs=1/step_size, axis=0)
                        self.processed_results[:, idx] = p
                    self.processed_results.index = f


            if __name__ == "__main__":
                my_worker = MyWorker()
                my_worker.worker_init()

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        self.processed_results = pd.DataFrame(data=None, columns=self.results.columns)
        for idx, data in self.results.iteritems():

            # Add post customized post processing of 'data' here
            self.processed_results.loc[:, idx] = data

    def worker_test(self):
        """
        Calls the worker_postprocessing script on a dummy result and a dummy result map

        self.result_map dummy looks as follows:

                       param_0  param_1
            circuit_0        0        9
            circuit_1        1        8
            circuit_2        2        7
            circuit_3        3        6
            circuit_4        4        5
            circuit_5        5        4
            circuit_6        6        3
            circuit_7        7        2
            circuit_8        8        1
            circuit_9        9        0

        self.results dummy looks as follows:

          circuit_0 circuit_1 circuit_2  ... circuit_7 circuit_8 circuit_9
                pop       pop       pop  ...       pop       pop       pop
            out_var   out_var   out_var  ...   out_var   out_var   out_var
        0       0.0       1.0       2.0  ...       7.0       8.0       9.0
        1      10.0      11.0      12.0  ...      17.0      18.0      19.0
        2      20.0      21.0      22.0  ...      27.0      28.0      29.0
        3      30.0      31.0      32.0  ...      37.0      38.0      39.0
        4      40.0      41.0      42.0  ...      47.0      48.0      49.0
        5      50.0      51.0      52.0  ...      57.0      58.0      59.0
        6      60.0      61.0      62.0  ...      67.0      68.0      69.0
        7      70.0      71.0      72.0  ...      77.0      78.0      79.0
        8      80.0      81.0      82.0  ...      87.0      88.0      89.0
        9      90.0      91.0      92.0  ...      97.0      98.0      99.0

        Returns
        -------

        """
        # Create Dummies
        circuit_names = [f'circuit_{i}' for i in range(10)]

        grid = {'param_0': list(range(10)),
                'param_1': list(range(10))[::-1]}

        dummy_grid = linearize_grid(grid, permute=False)
        dummy_grid.index = circuit_names

        pop = ['pop'] * 10
        out_var = ['out_var'] * 10

        dummy_mi = pd.MultiIndex.from_arrays([circuit_names, pop, out_var])
        dummy_results = pd.DataFrame(np.arange(100.0).reshape((10, 10)), columns=dummy_mi)

        self.result_map = dummy_grid
        self.results = dummy_results
        self.processed_results = pd.DataFrame(data=None, columns=self.results.columns)

        # Run post processing method
        self.worker_postprocessing()

        print('Test successful, no errors occurred!')

    @staticmethod
    def _limit_resources(cpu_lim=True, memory_lim=True, nproc_lim=False, time_lim=True):
        import resource

        # manage niceness level
        if cpu_lim:
            if type(cpu_lim) is bool:
                os.nice(10)
            else:
                os.nice(cpu_lim)

        # manage other resources
        limits = [memory_lim, nproc_lim, time_lim]
        resources = [resource.RLIMIT_AS, resource.RLIMIT_NPROC, resource.RLIMIT_CPU]
        default_limits = [4e9, 128, 3600]

        for limit, r, def_lim in zip(limits, resources, default_limits):
            if limit:
                _, hard = resource.getrlimit(r)
                if type(limit) is bool:
                    if def_lim < hard:
                        resource.setrlimit(r, (def_lim, hard))
                else:
                    if limit < hard:
                        resource.setrlimit(r, (limit, hard))


if __name__ == "__main__":
    cgs_worker = ClusterWorkerTemplate()
    # cgs_worker.worker_test()
    cgs_worker.worker_init()


