
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
from typing import Optional, Union

# system imports
import os
import sys
import json
import time as t
import glob
import getpass
import h5py
import paramiko
from shutil import copy2
from pathlib import Path
from datetime import datetime
from threading import Thread, currentThread, RLock

# pyrates internal imports
from pyrates.backend import ComputeGraph
from pyrates.frontend import CircuitTemplate
from pyrates.ir.circuit import CircuitIR

# meta infos
__author__ = "Christoph Salomon, Richard Gast"
__status__ = "development"


def grid_search(circuit_template: Union[CircuitTemplate, str], param_grid: Union[dict, pd.DataFrame], param_map: dict, dt: float, simulation_time: float,
                inputs: dict, outputs: dict, sampling_step_size: Optional[float] = None,
                permute_grid: bool = False, init_kwargs: dict = None, **kwargs) -> pd.DataFrame:
    """Function that runs multiple parametrizations of the same circuit in parallel and returns a combined output.

    Parameters
    ----------
    circuit_template
        Path to the circuit template.
    param_grid
        Key-value pairs for each circuit parameter that should be altered over different circuit parametrizations.
    param_map
        Key-value pairs that map the keys of param_grid to concrete circuit variables.
    dt
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
    kwargs
        Additional keyword arguments passed to the `:class:ComputeGraph` initialization.



    Returns
    -------
    pd.DataFrame
        Simulation results stored in a multi-index data frame where each index lvl refers to one of the parameters of
        param_grid.

    """

    # load template if necessary
    if type(circuit_template) is str:
        circuit_template = CircuitTemplate.from_yaml(circuit_template)

    # linearize parameter grid if necessary
    if type(param_grid) is dict:
        param_grid = linearize_grid(param_grid, permute_grid)

    # create multi-index for later results storage
    param_keys = list(param_grid.keys())
    multi_idx = [param_grid[key].values for key in param_keys]
    n_iters = len(multi_idx[0])
    outs = []
    out_names = list(outputs.keys())
    for out_name in out_names:
        outs += [out_name] * n_iters
    multi_idx = [list(idx) * len(out_names) for idx in multi_idx]
    multi_idx = multi_idx + [outs]
    index = pd.MultiIndex.from_arrays(multi_idx, names=param_keys + ["out_var"])
    index = pd.MultiIndex.from_tuples(list(set(index)), names=param_keys + ["out_var"])

    # assign parameter updates to each circuit and combine them to unconnected network
    circuit = CircuitIR()
    circuit_names = []
    param_keys = index.names
    results_indices = []
    i = 0
    for idx in index.values:
        if idx[:-1] not in results_indices:
            results_indices.append(idx[:-1])
            new_params = {}
            for key, val in zip(param_keys, idx):
                if key in param_grid:
                    new_params[key] = val
            circuit_tmp = circuit_template.apply()
            circuit_names.append(f'{circuit_tmp.label}_{i}')
            circuit_tmp = adapt_circuit(circuit_tmp, new_params, param_map)
            circuit.add_circuit(circuit_names[-1], circuit_tmp)
            i += 1

    # create backend graph
    if not init_kwargs:
        init_kwargs = {}
    net = ComputeGraph(circuit, dt=dt, **init_kwargs)

    # adjust input of simulation to combined network
    for inp_key, inp in inputs.items():
        inputs[inp_key] = np.tile(inp, (1, len(circuit_names)))

    # adjust output of simulation to combined network
    nodes = list(circuit_template.apply().nodes)
    out_nodes = []
    for out_key in out_names:
        out = outputs.pop(out_key)
        out_nodes.append(out)
        if out[0] in nodes:
            out_tmp = list(out)
            out_tmp[0] = out_tmp[0].split('.')[0]
            outputs[out_key] = tuple(out_tmp)

    # simulate the circuits behavior
    results = net.run(simulation_time=simulation_time,
                      inputs=inputs,
                      outputs=outputs,
                      sampling_step_size=sampling_step_size,
                      **kwargs)    # type: pd.DataFrame
    if 'profile' in kwargs:
        results, duration, memory = results

    # transform results into long-form dataframe with changed parameters as columns
    results_final = pd.DataFrame(columns=index, data=np.zeros_like(results.values), index=results.index)
    for out_name, out_info in zip(out_names, out_nodes):
        node, op, var = out_info
        for node_name, params in zip(circuit_names, results_indices):
            key = params + (out_name,)
            idx = list(net.get_var(f"{node_name}/{node}", op, var, retrieve=False).values())
            results_final[key] = results[out_name].iloc[:, idx]

    if 'profile' in kwargs:
        return results_final, duration, memory
    return results_final


class ClusterCompute:
    def __init__(self, nodes: list, compute_dir=None):
        """Create new ClusterCompute instance object with unique compute ID

        Creates a compute directory for the ClusterCompute instance, either in the specified path or as a default
        folder in the current working directory of the executing script.
        Creates a logfile in ComputeDirectory/Logs and tees all future stdout and stderr of the executing script
        to this file
        Connects to all nodes via SSH and saves the corresponding paramiko client in self.clients

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
            try:
                client["paramiko_client"].close()
            except TypeError:
                pass

    def cluster_connect(self, nodes: list):
        """Create SSH connections to all nodes in the list, respectively

        Connect to all nodes in the given list via SSH, using ssh_connect(). Adds a dictionary for each node to the
        class internal 'clients' list
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

    def spawn_thread(self, client, thread_kwargs):
        """

        Parameters
        ----------
        client
        thread_kwargs

        Returns
        -------

        """
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
        print("")

        hw = {
            'cpu': cpu,
            'num_cpu_cores': num_cpu_cores,
            'cpu_min': cpu_min,
            'cpu_max': cpu_max,
            'total_mem': total_mem
        }
        return hw

    @staticmethod
    def ssh_connect(node, username, password=None, print_status=True):
        """Connect to a host via SSH and return a respective paramiko.SSHClient

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
    def __init__(self, nodes, compute_dir=None):
        super().__init__(nodes, compute_dir)

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

    def run(self, circuit_template: str, params: Union[dict, pd.DataFrame], param_map: dict, dt: float, simulation_time: float,
            inputs: dict, outputs: dict, sampling_step_size: float, chunk_size: (int, list),
            worker_env: str, worker_file: str, result_kwargs: dict = {}, config_kwargs: dict = {},
            add_template_info: bool = False, permute: bool = False, **kwargs) -> str:
        """Run multiple instances of grid_search simultaneously on different workstations in the compute cluster

        ClusterGridSearch requires all necessary data (worker script, worker environment, compute directory)
        to be stored on a server. After a connection is established, all data on the remote workers has to be accessible
        in the same way as it would be on the master worker (same root directory, same filepaths, etc.)
        Paths inside the run function have to be adjusted respectively.

        FTP implementation is planned in future versions for direct file transfer between the master and the workers.

        Parameters
        ----------
        circuit_template
            Path to the circuit template.
        params
            Dictionary containing lists of parameters to create the parameter grid from.
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
        permute
            If true, all combinations of the parameter values in params will be created.
        chunk_size
            Number of parameter combinations computed simultaneously on one worker
        worker_env
            Python executable inside an environment in which the remote worker script is called.
        worker_file
            Python script that will be executed by each remote worker.
        result_kwargs
            Key-value pairs that will be added to the result file's 'AdditionalData' dataset
        config_kwargs
            Key-value pairs that will be added to the config file.
        add_template_info
            If true, all operator templates and its variables are copied from the yaml file to a dedicated folder in the
            result file

        Returns
        -------
        str
            .hdf5 file containing the computation results as DataFrame in dataset '/Results/...'
        """

        t_total = t.time()

        print("")

        # Prepare parameter grid
        ########################
        print("***PREPARING PARAMETER GRID***")
        t0 = t.time()

        # Create DataFrame from param dictionary
        if isinstance(params, dict):
            param_grid = linearize_grid(params, permute=permute)
        else:
            param_grid = params

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
        if not self.check_key_consistency(params, param_map):
            print("Terminating execution!")
            return ""
        print(f'Done! Elapsed time: {t.time() - t0:.3f} seconds')

        print("")

        # Create config file
        #####################
        print("***CREATING CONFIG FILE***")
        t0 = t.time()
        config_idx = 0
        while os.path.exists(f'{self.config_dir}/DefaultConfig_{config_idx}.json'):
            config_idx += 1
        config_file = f'{self.config_dir}/DefaultConfig_{config_idx}.json'
        self.create_cgs_config(fp=config_file,
                               circuit_template=circuit_template,
                               param_map=param_map,
                               dt=dt,
                               simulation_time=simulation_time,
                               inputs=inputs,
                               outputs=outputs,
                               sampling_step_size=sampling_step_size,
                               add_kwargs=config_kwargs)
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
            for key, value in params.items():
                file.create_dataset(f'/ParameterGrid/Keys/{key}', data=value)
            for key, value in result_kwargs.items():
                file.create_dataset(f'/AdditionalData/{key}', data=value)
            file.create_dataset(f'/Config/config_file', data=config_file)
            file.create_dataset(f'/Config/circuit_template', data=circuit_template)
            file.create_dataset(f'/Config/simulation_time', data=simulation_time)
            file.create_dataset(f'/Config/dt', data=dt)
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

        # TODO: Copy worker environment, worker file and configuration file to the worker, if necessary

        # Start cluster computation
        ###########################
        print("***STARTING CLUSTER COMPUTATION***")
        # Spawn threads to control each node connection and start computation
        threads = [self.spawn_thread(client, thread_kwargs) for client in self.clients]
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
        with pd.HDFStore(temp_res_files[0], "r") as store:
            for key in store.keys():
                res_dict[key] = []

        # Concatenate results from each temporary result file
        for file in temp_res_files:
            with pd.HDFStore(file, "r") as store:
                for idx, key in enumerate(store.keys()):
                    res_dict[key].append(store[key])

        # Create DataFrame for each output variable and write to global result file
        with pd.HDFStore(global_res_file, "a") as store:
            for key, value in res_dict.items():
                if len(value) > 0:
                    df = pd.concat(value, axis=1)
                    store.put(key=f'/Results/{key}', value=df)

        # Delete temporary local result files
        for file in temp_res_files:
            os.remove(file)

        print(f'Elapsed time: {t.time()-t0:.3f} seconds')
        print(f'Find results in: {grid_res_dir}')
        print("")

        working_grid.to_csv(f'{os.path.dirname(grid_file)}/{grid_name}_{self.compute_id}_ResultStatus.csv', index=True)

        # self.__del__()
        return global_res_file

    def thread_master(self, client: dict, thread_kwargs: dict):
        """Function that is executed by each thread to schedule computations on the respective worker

        Parameters
        ----------
        client
            Dictionary containing all information about the remote worker that is tied to the current thread
        thread_kwargs
            Dictionary containing all kwargs that are passed to the thread function

        Returns
        -------

        """
        thread_name = currentThread().getName()
        connection_lost = False
        connection_lost_counter = 0

        # Get client information
        pm_client = client["paramiko_client"]
        logfile = client["logfile"]
        chunk_size = client["chunk_size"]

        # Get keyword arguments
        config_file = thread_kwargs["config_file"]
        working_grid = thread_kwargs["working_grid"]
        grid_name = thread_kwargs["grid_name"]
        grid_res_dir = thread_kwargs["grid_res_dir"]
        worker_env = thread_kwargs["worker_env"]
        worker_file = thread_kwargs["worker_file"]

        # Prepare worker command
        command = f'{worker_env} {worker_file}'

        # Create folder to save local subgrids to
        subgrid_dir = f'{self.grid_dir}/Subgrids/{grid_name}/{thread_name}'
        os.makedirs(subgrid_dir, exist_ok=True)
        subgrid_idx = 0

        # Start scheduler
        #################
        while True:

            # Temporarily disable thread switching
            with self.lock:

                # Fetch grid chunks
                ####################

                # Get all parameters that haven't been successfully computed yet
                remaining_params = working_grid.loc[working_grid["status"] == "unsolved"]
                if remaining_params.empty:
                    print(f'[T]\'{thread_name}\': No more parameter combinations available!')
                    break
                else:
                    # Find chunk index of first value in remaining params and fetch all parameter combinations with the
                    # same chunk index
                    param_idx, chunk_idx = self._fetch_index(remaining_params, chunk_size, working_grid)
                    working_grid.loc[param_idx, "status"] = "pending"

                    print(f'[T]\'{thread_name}\': Fetching {len(param_idx)} indices: ', end="")
                    print(f'[{param_idx[0]}] - [{param_idx[-1]}]')

                    # Create parameter sub-grid
                    ###########################
                    subgrid_fp = f'{subgrid_dir}/{thread_name}_Subgrid_{subgrid_idx}.h5'
                    subgrid_df = working_grid.iloc[param_idx]
                    subgrid_df.to_hdf(subgrid_fp, key="subgrid")
                    subgrid_idx += 1

                    # TODO: Copy subgrid to worker if necessary

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
                        stdin, stdout, stderr = pm_client.exec_command(command +
                                                                       f' --config_file={config_file}'
                                                                       f' --subgrid={subgrid_fp}'
                                                                       f' --local_res_file={local_res_file}'
                                                                       f' &>> {logfile}',  # redirect and append stdout
                                                                                           # and stderr to logfile
                                                                       get_pty=True)       # execute in pseudoterminal
                    except paramiko.ssh_exception.SSHException as e:
                        # SSH connection has been lost
                        # (remote machine shut down, ssh connection has been killed manually, ...)
                        print(f'[T]\'{thread_name}\': ERROR: {e}')
                        working_grid.at[param_idx, "status"] = "unsolved"
                        connection_lost = True

            # Lock released, thread switching enabled

            # Try to reconnect to host if connection has been lost
            ######################################################
            if connection_lost and connection_lost_counter < 2:
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
            exit_status = stdout.channel.recv_exit_status()

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
                            if working_grid.loc[row]["err_count"] < 2:
                                working_grid.at[row, "status"] = "unsolved"
                                working_grid.at[row, "err_count"] = working_grid.iloc[row]["err_count"] + 1
                            else:
                                working_grid.at[row, "status"] = "failed"

                # Remote execution was interrupted
                else:
                    print(f'[T]\'{thread_name}\': Remote computation failed with exit status {exit_status}')
                    connection_lost_counter += 1
                    for row in param_idx:
                        if working_grid.loc[row]["err_count"] < 4:
                            working_grid.at[row, "status"] = "unsolved"
                            working_grid.at[row, "err_count"] = working_grid.iloc[row]["err_count"] + 1
                        else:
                            working_grid.at[row, "status"] = "failed"
                    if connection_lost_counter >= 2:
                        print('Excluding worker from pool')
                        return

            # Lock released, thread switching enabled
        # End of scheduler loop

        # TODO: Close pty on remote machine?

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
                working_grid.loc[i, 'chunk_idx'] = self.chunk_idx
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
    def check_key_consistency(param_grid: dict, param_map: dict):
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
            print("Not all parameter grid keys found in parameter map")
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
                          inputs: dict, outputs: dict, sampling_step_size: float, add_kwargs: dict):
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
        add_kwargs

        Returns
        -------

        """
        config_dict = {
            "circuit_template": circuit_template,
            "param_map": param_map,
            "dt": dt,
            "simulation_time": simulation_time,
            "outputs": outputs,
            "sampling_step_size": sampling_step_size,
        }
        if inputs:
            config_dict["inputs"] = {str(*inputs.keys()): list(*inputs.values())}
        for key, value in add_kwargs.items():
            config_dict[key] = value
        with open(fp, "w") as f:
            json.dump(config_dict, f, indent=2)

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
    else:
        vals, keys = [], []
        for key, val in grid.items():
            vals.append(val)
            keys.append(key)
        new_grid = np.stack(np.meshgrid(*tuple(vals)), -1).reshape(-1, len(grid))
        return pd.DataFrame(new_grid, columns=keys)


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
        for op, var in param_map[key]['var']:
            nodes = param_map[key]['nodes'] if 'nodes' in param_map[key] else []
            edges = param_map[key]['edges'] if 'edges' in param_map[key] else []
            for node in nodes:
                if op in circuit.nodes[node]['node'].op_graph.nodes:
                    circuit.nodes[node]['node'].op_graph.nodes[op]['variables'][var]['value'] = float(val)
            for source, target, edge in edges:
                if op in circuit.edges[source, target, edge]:
                    circuit.edges[source, target, edge][op][var] = float(val)
                elif var in circuit.edges[source, target, edge]:
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
