
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
from typing import Optional
import paramiko

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

# pyrates internal imports
from pyrates.backend import ComputeGraph
from pyrates.frontend import CircuitTemplate
from pyrates.ir.circuit import CircuitIR

# meta infos
__author__ = "Christoph Salomon, Richard Gast"
__status__ = "development"


def grid_search(circuit_template: str, param_grid: dict, param_map: dict, dt: float, simulation_time: float,
                inputs: dict, outputs: dict, sampling_step_size: Optional[float] = None,
                permute_grid: bool = False, **kwargs) -> pd.DataFrame:
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
        Simulation results stored in a multi-index data frame where each index level refers to one of the parameters of
        param_grid.

    """

    # linearize parameter grid if necessary
    if type(param_grid) is dict:
        param_grid = linearize_grid(param_grid, permute_grid)

    # assign parameter updates to each circuit and combine them to unconnected network
    circuit = CircuitIR()
    circuit_names = []
    param_info = []
    param_split = "__"
    val_split = "--"
    comb = "_"
    for n in range(param_grid.shape[0]):
        circuit_tmp = CircuitTemplate.from_yaml(circuit_template).apply()
        circuit_names.append(f'{circuit_tmp.label}_{n}')
        circuit_tmp = adapt_circuit(circuit_tmp, param_grid.iloc[n, :], param_map)
        circuit.add_circuit(circuit_names[-1], circuit_tmp)
        param_names = list(param_grid.columns.values)
        param_info_tmp = [f"{param_names[i]}{val_split}{val}" for i, val in enumerate(param_grid.iloc[n, :])]
        param_info.append(param_split.join(param_info_tmp))

    # create backend graph
    net = ComputeGraph(circuit, dt=dt, **kwargs)

    # adjust input of simulation to combined network
    for inp_key, inp in inputs.items():
        inputs[inp_key] = np.tile(inp, (1, len(circuit_names)))

    # adjust output of simulation to combined network
    nodes = list(CircuitTemplate.from_yaml(circuit_template).apply().nodes)
    out_names = []
    for out_key, out in outputs.copy().items():
        outputs.pop(out_key)
        out_names_tmp = []
        if out[0] in nodes:
            for i, name in enumerate(param_info):
                out_tmp = list(out)
                out_tmp[0] = f'{circuit_names[i]}/{out_tmp[0]}'
                outputs[f'{name}{param_split}out_var{val_split}{out_key}{comb}{out[0]}'] = tuple(out_tmp)
                out_names_tmp.append(f'{out_key}{comb}{out[0]}')
        elif out[0] == 'all':
            for node in nodes:
                for i, name in enumerate(param_info):
                    out_tmp = list(out)
                    out_tmp[0] = f'{circuit_names[i]}/{node}'
                    outputs[f'{name}{param_split}out_var{val_split}{out_key}{comb}{node}'] = tuple(out_tmp)
                    out_names_tmp.append(f'{out_key}{comb}{node}')
        else:
            node_found = False
            for node in nodes:
                if out[0] in node:
                    node_found = True
                    for i, name in enumerate(param_info):
                        out_tmp = list(out)
                        out_tmp[0] = f'{circuit_names[i]}/{node}'
                        outputs[f'{name}{param_split}out_var{val_split}{out_key}{comb}{node}'] = tuple(out_tmp)
                        out_names_tmp.append(f'{out_key}{comb}{node}')
            if not node_found:
                raise ValueError(f'Invalid output identifier in output: {out_key}. '
                                 f'Node {out[0]} is not part of this network')
        out_names += list(set(out_names_tmp))

    # simulate the circuits behavior
    results = net.run(simulation_time=simulation_time,
                      inputs=inputs,
                      outputs=outputs,
                      sampling_step_size=sampling_step_size)    # type: pd.DataFrame

    # transform results into long-form dataframe with changed parameters as columns
    multi_idx = [param_grid[key].values for key in param_grid.keys()]
    n_iters = len(multi_idx[0])
    outs = []
    for out_name in out_names:
        outs += [out_name] * n_iters
    multi_idx = [list(idx) * len(out_names) for idx in multi_idx]
    multi_idx.append(outs)
    index = pd.MultiIndex.from_arrays(multi_idx, names=list(param_grid.keys()) + ["out_var"])
    index = pd.MultiIndex.from_tuples(list(set(index)), names=list(param_grid.keys()) + ["out_var"])
    results_final = pd.DataFrame(columns=index, data=np.zeros_like(results.values), index=results.index)
    for col in results.keys():
        params = col[0].split(param_split)
        indices = [None] * len(results_final.columns.names)
        for param in params:
            var, val = param.split(val_split)[:2]
            idx = list(results_final.columns.names).index(var)
            try:
                indices[idx] = float(val)
            except ValueError:
                indices[idx] = val
        results_final.loc[:, tuple(indices)] = results[col].values

    return results_final


class ClusterCompute(object):
    def __init__(self, nodes, compute_dir=None):
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

    def run(self, thread_kwargs: dict, **kwargs):
        """Start a thread for each connected client. Each thread executes the thread_master() function

        Each thread and therefor each instance of the thread_master() function is responsible for the communication
        with one worker node in the cluster, respectively.
        Stops execution of the outside script until all threads have finished.
        Can be called multiple times from the same ClusterCompute instance

        Params
        ------
        kwargs

        Returns
        -------

        """

        t0 = t.time()

        # ! Insert preprocessing of input data here !

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
        try:
            # Using kerberos authentication
            client.connect(node, username=username, gss_auth=True, gss_kex=True)
            print(f'\'{node}\': Connection established!')
            return client
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

        self.res_file_idx = 0
        self.res_collection = {}

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

        # Reset results organizer
        #########################
        # Index to iterate over all temporary result files to create a single result file from the res_collection
        self.res_file_idx = 0
        self.res_collection = {}

        # Retrieve keyword arguments
        ############################
        config_file = kwargs.pop("config_file")
        params = kwargs.pop("param_grid")
        permute = kwargs.pop("permute")
        chunk_size = kwargs.pop("chunk_size")

        print("")

        # Prepare parameter grid
        ########################
        print("***PREPARING PARAMETER GRID***")
        t0 = t.time()

        # Convert parameter from dict to DataFrame. Permute if necessary
        param_grid_raw = linearize_grid(params, permute=permute)

        # Add 'status' and 'worker' column to parameter grid
        param_grid, grid_file, grid_name = self.prepare_grid(param_grid_raw)

        print(f'Done. Elapsed time: {t.time()-t0:.3f} seconds')

        print("")

        # Check key consistency
        #######################
        print("***CHECKING PARAMETER GRID AND MAP FOR CONSISTENCY***")

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

        print("")

        # Prepare result file
        #####################
        print("***PREPARING RESULT FILE***")
        t0 = t.time()

        # Create result directory and result file for current parameter grid
        grid_res_dir = f'{self.res_dir}/{grid_name}'
        os.makedirs(grid_res_dir, exist_ok=True)
        global_res_file = f'{grid_res_dir}/CGS_result_{grid_name}.h5'

        # Write grid and config information to result file
        with h5py.File(global_res_file, 'w') as file:
            # file.create_dataset(f'/ParameterGrid/Grid', data=param_grid_raw.values)
            for key, value in params.items():
                file.create_dataset(f'/ParameterGrid/Keys/{key}', data=value)
            file.create_dataset(f'/Config/circuit_template', data=param_dict['circuit_template'])
            file.create_dataset(f'/Config/simulation_time', data=param_dict['simulation_time'])
            file.create_dataset(f'/Config/dt', data=param_dict['dt'])
            file.create_dataset(f'/Config/sampling_step_size', data=param_dict['sampling_step_size'])
        param_grid_raw.to_hdf(global_res_file, key='/ParameterGrid/Grid_df')

        print(f'Done. Elapsed time: {t.time() - t0:.3f} seconds')

        print("")

        # Prepare chunk sizes
        #####################
        print("***PREPARING CHUNK SIZES***")
        t0 = t.time()

        # Prepare the chunk sizes
        self.prepare_chunks(grid_len=len(param_grid), chunk_size=chunk_size)

        print(f'Done. Elapsed time: {t.time() - t0:.3f} seconds')

        print("")

        # Update thread keywords dictionary
        ###################################
        thread_kwargs["param_grid"] = param_grid
        thread_kwargs["grid_name"] = grid_name
        thread_kwargs["grid_res_dir"] = grid_res_dir
        thread_kwargs["config_file"] = config_file
        thread_kwargs["global_res_file"] = global_res_file

        # Begin cluster computation
        ###########################
        print("***STARTING CLUSTER COMPUTATION***")

        # Spawn threads to control each node connections and start computation
        threads = [self.spawn_thread(client, thread_kwargs) for client in self.clients]
        for t_ in threads:
            t_.join()

        print("")

        print(f'Cluster computation finished. Elapsed time: {t.time() - t_total:.3f} seconds')

        print("")

        # Creating global result file
        ###############################
        print(f'***CREATING GLOBAL RESULT FILE***')
        t0 = t.time()

        res_lst = []
        for i in range(self.res_file_idx):
            res_lst.append(self.res_collection.pop(i))
        df = pd.concat(res_lst, axis=1)

        print(f'Writing data to global result file...', end="")
        with pd.HDFStore(global_res_file, "a") as store:
            store.put(key='/Results/r_E0_df', value=df)
        print("done")

        print(f'Deleting temporary result files...', end="")
        for temp_file in glob.glob(f'{grid_res_dir}/*_temp*.h5'):
            os.remove(temp_file)
        print("done")

        print(f'Elapsed time: {t.time()-t0:.3f} seconds')
        print(f'Find results in: {grid_res_dir}/')

        print("")

        param_grid.to_csv(f'{os.path.dirname(grid_file)}/{grid_name}_{self.compute_id}_ResultStatus.csv', index=True)

        return global_res_file, grid_file

    def thread_master(self, client, thread_kwargs: dict):
        thread_name = currentThread().getName()

        # Get client information
        pm_client = client["paramiko_client"]
        logfile = client["logfile"]
        num_params = client["num_params"]

        # Get keyword arguments
        config_file = thread_kwargs["config_file"]
        param_grid = thread_kwargs["param_grid"]
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

                # Fetch grid indices
                ####################
                param_idx = self.fetch_param_idx(param_grid, num_params=num_params)
                if param_idx.empty:
                    print(f'[T]\'{thread_name}\': No more parameter combinations available!')
                    # Exit while loop
                    break
                else:
                    print(f'[T]\'{thread_name}\': Fetching {len(param_idx)} indices: ', end="")
                    print(f'[{param_idx[0]}] - [{param_idx[-1]}]')

                    # Create parameter sub-grid
                    ###########################
                    subgrid_fp = f'{subgrid_dir}/{thread_name}_Subgrid_{subgrid_idx}.h5'
                    subgrid_df = param_grid.iloc[param_idx]
                    subgrid_df.to_hdf(subgrid_fp, key='Data')
                    subgrid_idx += 1

                    # Create temporary result file for current subgrid
                    ##################################################
                    tmp_res_idx = self.res_file_idx
                    idx_min = np.amin(param_idx.values)
                    idx_max = np.amax(param_idx.values)
                    local_res_file = f'{grid_res_dir}/CGS_result_{grid_name}_idx_{idx_min}-{idx_max}_temp.h5'
                    # TODO: Write indices to local_res_file
                    self.res_file_idx += 1

                    # Execute worker script on the remote host
                    ##########################################
                    t0 = t.time()
                    print(f'[T]\'{thread_name}\': Starting remote computation...')

                    stdin, stdout, stderr = pm_client.exec_command(command +
                                                                   f' --config_file={config_file}'
                                                                   f' --subgrid={subgrid_fp}'
                                                                   f' --local_res_file={local_res_file}'
                                                                   # redirect and append stdout and stderr to logfile:
                                                                   f' &>> {logfile}',
                                                                   get_pty=True)
            # Lock released, thread switching enabled

            # Wait for remote computation to finish
            #######################################
            status = stdout.channel.recv_exit_status()
            print(f'[T]\'{thread_name}\': Remote computation finished. Elapsed time: {t.time()-t0:.3f} seconds')

            # Load results from result file to working memory
            #################################################
            # t0 = t.time()
            print(f'[T]\'{thread_name}\': Updating grid status')

            with self.lock:
                try:
                    # TODO: Read results from Port (Socket) ?
                    local_results = pd.read_hdf(local_res_file, key='Data')
                    # for col in range(local_results.shape[1]):
                    #     local_result = local_results.iloc[:, col]
                    #     idx_label = local_result.name[:-1]
                    #     idx = param_grid[(param_grid_raw.values == idx_label).all(1)].index
                    #     if any(pd.isnull(local_result)):
                    #         param_grid.at[idx, 'status'] = 'unsolved'
                    #     else:
                    #         param_grid.at[idx, 'status'] = 'done'
                    # param_grid.at[param_grid['status'] == 'pending', 'status'] = 'unsolved'
                    param_grid.at[param_idx, 'status'] = 'done'
                    self.res_collection[tmp_res_idx] = local_results
                except KeyError:
                    # TODO: restart computation only twice before setting status to 'failed'
                    param_grid.at[param_idx, 'status'] = 'unsolved'
                    # os.remove(local_res_file)
                except FileNotFoundError:
                    param_grid.at[param_idx, 'status'] = 'unsolved'
                # TODO: What happens, if results are not added to res_collection, but the res_idx is still counted up?

            # Lock released, thread switching enabled
        # End of while loop
    # End of Thread master

    ########################
    # CGS Helper functions #
    ########################

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
            grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'
            # If grid_file already exist
            while os.path.exists(grid_file):
                grid_idx += 1
                grid_file = f'{self.grid_dir}/DefaultGrid_{grid_idx}.csv'
            # grid_frame.to_csv(grid_file, index=True, float_format='%g')
            # grid_frame.to_csv(grid_file, index=True)
            grid_frame.to_csv(grid_file, index=True)
            # Add status columns to grid
            if 'status' not in grid_frame.columns:
                grid_frame['status'] = 'unsolved'
            else:
                # Set rows with status 'failed' to 'unsolved' to compute them again
                unsolved_idx = grid_frame.index[grid_frame['status'] == "failed"]
                grid_frame.at[unsolved_idx, 'status'] = 'unsolved'
            # Add/reset worker-column
            grid_name = Path(grid_file).stem
            return grid_frame, grid_file, grid_name

        elif isinstance(param_grid_arg, pd.DataFrame):
            grid_frame = param_grid_arg.copy(deep=True)
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
            grid_name = Path(grid_file).stem
            return grid_frame, grid_file, grid_name

        else:
            print("Parameter grid unsupported format")
            # Stop computation
            return None

    def prepare_chunks(self, grid_len, chunk_size="dist_equal"):
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

    def fetch_param_idx(self, param_grid, lock=False, num_params=1, set_status=True):
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
            with self.lock:
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
    # if not os.path.exists(fp):
    config_dict = {
        "circuit_template": circuit_template,
        "param_map": param_map,
        "dt": dt,
        "simulation_time": simulation_time,
        "outputs": outputs,
        "sampling_step_size": sampling_step_size,
        "kwargs": kwargs
    }
    if inputs:
        config_dict["inputs"] = {str(*inputs.keys()): list(*inputs.values())}

    with open(fp, "w") as f:
        json.dump(config_dict, f, indent=2)
