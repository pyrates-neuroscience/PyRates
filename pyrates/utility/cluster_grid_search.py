
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
"""Functions for performing parameter grid simulations with pyrates models.
"""

# external imports
import pandas as pd
import numpy as np
import paramiko

# system imports
import json
import getpass
from threading import Thread, currentThread, RLock

# pyrates internal imports
from pyrates.utility.grid_search import linearize_grid
# from pyrates.backend import ComputeGraph
# from pyrates.frontend import CircuitTemplate
# from pyrates.ir.circuit import CircuitIR

# meta infos
__author__ = "Christoph Salomon"
__status__ = "development"


def create_cgs_config_file(config_fp, circuit_template, param_grid, param_map, dt, simulation_time, inputs,
                           outputs, sampling_step_size=None, permute_grid=False, **kwargs):
    """

    Parameters
    ----------
    config_fp
    circuit_template
    param_grid
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

    # setswitchinterval(1)

    if type(param_grid) is dict:
        # convert linear_grid from dict to pandas.DataFrame. Add status_flag later
        param_grid = linearize_grid(param_grid, permute_grid)

    # TODO: Eliminate extra brackets in inputs in config_file
    config_dict = {
        "circuit_template": circuit_template,
        "param_grid": param_grid.to_dict(),
        "param_map": param_map,
        "dt": dt,
        "simulation_time": simulation_time,
        "inputs": {str(*inputs.keys()): list(*inputs.values())},
        "outputs": outputs,
        "sampling_step_size": sampling_step_size,
        "permute_grid": permute_grid,
        "kwargs": kwargs
    }

    with open(config_fp, "w") as f:
        json.dump(config_dict, f, indent=2)


def cluster_grid_search(hosts, host_config, config_file, param_grid=None, **kwargs):
    """

    Parameters
    ----------
    hostnames
    config_file
    param_grid
    py_env
    worker
    kwargs

    Returns
    -------

    """
    # Create parameter grid from config_file.json if not specified
    if not param_grid:
        try:
            print(f'Loading config file: {config_file}... ', end="")
            with open(config_file, "r") as file:
                param_dict = json.load(file)
                print("done!")
                print("Creating parameter grid... ", end="")
                try:
                    # Create a pandas.DataFrame() from param_grid{} in param_dict{}
                    param_grid = pd.DataFrame(param_dict["param_grid"])
                    # Add 'status' key for scheduler
                    param_grid['status'] = 'unsolved'
                    print("done!")
                except KeyError as err:
                    # If config_file does not contain a key named 'param_grid':
                    print("\nKeyError:", err)
                    return
        except IOError as err:
            # If config_file does not exist:
            print("\nIOError:", err)
            return
    else:
        print("Linearizing parameter grid...")
        param_grid = linearize_grid(param_grid, permute=True)
        # Add 'status' key for scheduler
        param_grid['status'] = 'unsolved'

    # TODO: Implement threadpool instead of single threads in a loop
    # TODO: Implement asynchronous computation instead of multithreading
    # Start a thread for each host to handle the SSH-connection
    print("Starting threads...")
    password = getpass.getpass(
        prompt='Enter password:', stream=None)

    lock = RLock()
    threads = []
    for host in hosts['hostnames']:
        threads.append(spawn_thread(host=host,
                                    host_cmd={hosts['host_env'], hosts['host_file']},
                                    param_grid=param_grid,
                                    config_file=config_file,
                                    password=password,
                                    lock=lock))

    # Wait for all threads to finish
    for t in threads:
        t.join()

    print(param_grid)
    # return results
    # TODO: Create log file


def spawn_thread(host, host_cmd, param_grid, config_file, password, lock):
    t = Thread(
        name=host,
        target=thread_master,
        args=(host, host_cmd, param_grid, config_file, password, lock)
    )
    t.start()
    return t


def thread_master(host, host_cmd, param_grid, config_file, password, lock):
    # Optional via lock: Make sure to connect to every host before starting a computation
    # lock.acquire()
    thread_name = currentThread().getName()

    # create SSH Client/Channel
    # TODO: Implement connection with key-files and no password
    client = create_ssh_connection(host,
                                   username=getpass.getuser(),
                                   password=password)
    # lock.release()

    # Check if create_ssh_connection() didn't return 0
    if client:
        # Check if 'status'-key is present in param_grid
        if not fetch_param_idx(param_grid, set_status=False).isnull():
            # Command to send to remote host
            command = host_cmd['host_env'] + ' ' + host_cmd['host_file']
            # TODO: Copy environment, script and config to shared directory or local directory on the host
            # -> Change paths of env and workerfile respectively

            # TODO: Call exec_command only once and communicate with it via stdin inside the while loop
            # stdin.write()
            # stdin.flush()

            # Check for available parameters to fetch
            while not fetch_param_idx(param_grid, set_status=False).empty:
                lock.acquire()

                param_idx = fetch_param_idx(param_grid)
                param_grid_arg = param_grid.iloc[param_idx]
                print(f'\'{thread_name}\': fetching index {param_idx}...')

                # TODO: Pass param_grid_arg as additional argument to exec_command()
                stdin, stdout, stderr = client.exec_command(command +
                                                            f' --param_grid_arg="{param_grid_arg.to_dict()}"'
                                                            f' --config_file="{config_file}"',
                                                            get_pty=True)
                # Pass lock to another thread after a remote computation is started
                lock.release()

                exit_status = stdout.channel.recv_exit_status()

                for line in iter(stdout.readline, ""):
                    print(f'\'{thread_name}\': {line}', end="")

                print(f'\'{thread_name}\': done!')

                # TODO: Create result file and concatenate the intermediate results directly to this file
                # TODO: Change status from current param_idx in param_grid from 'pending' to 'done'
                # result = pd.read_csv(stdout)

            # print(fetch_param_idx(param_grid, set_status=False).empty)
            # print(param_grid)
        else:
            # If no key named 'status' in param_grid:
            print(f'\'{host}\': "No key named \'status\' in param_grid')

        client.close()
        # return result


def create_ssh_connection(host, username, password):
    """Connect to a host via SSH

    Parameters
    ----------
    host
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
    print(f'Attempting to connect to host \'{host}\'...')
    try:
        client.connect(host, username=username, password=password)
        print(f'\'{host}\': Connection established')
        return client
        # return client.invoke_shell()
    except paramiko.ssh_exception.NoValidConnectionsError as err:
        print(f'\'{host}\': ', err)
        return 0
    except paramiko.ssh_exception.AuthenticationException as err:
        print(f'\'{host}\': ', err)
        return 0
    except paramiko.ssh_exception.SSHException as err:
        print(f'\'{host}\': ', err)
        return 0
    except IOError as err:
        print(f'\'{host}\': ', err)
        return 0


def fetch_param_idx(param_grid, num_params=1, set_status=True):
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
    try:
        # Get the first num_params row indices of lin_grid who's 'status' keys equal 'unsolved'
        param_idx = param_grid.loc[param_grid['status'] == 'unsolved'].index[:num_params]
    except KeyError:
        # print("DataFrame doesn't contain a key named 'status'")
        return pd.Index([np.nan])
    if set_status:
        param_grid.at[param_idx, 'status'] = 'pending'

    return param_idx
    # To access the selected data use fetched_params = lin_grid.iloc[param_idx]

