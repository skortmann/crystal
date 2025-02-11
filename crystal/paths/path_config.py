#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of path_config

Copyright (c) by Institute for High Voltage Equipment and Grids, 
Digitalization and Energy Economics (IAEW), RWTH Aachen University, 
28.01.2025, s.kortmann. All rights reserved.
"""
import pathlib

import yaml
import os
from datetime import datetime
from typing import Any, Dict

class Scenario:
    """
    A class to define and manage parameters for a scenario run.

    Attributes:
        parameters (dict): A dictionary containing all scenario parameters.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Scenario with parameters.

        Args:
            **kwargs: Arbitrary keyword arguments representing scenario parameters.
        """
        self.parameters = kwargs

    def update_parameters(self, **kwargs):
        """
        Update the parameters of the scenario.

        Args:
            **kwargs: Arbitrary keyword arguments representing new or updated scenario parameters.
        """
        self.parameters.update(kwargs)

    def to_yaml(self, file_path: str):
        """
        Export the scenario parameters to a YAML configuration file.

        Args:
            file_path (str): The file path to save the YAML configuration.
        """
        with open(file_path, 'w') as yaml_file:
            yaml.dump(self.parameters, yaml_file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, file_path: str):
        """
        Create a Scenario instance from a YAML configuration file.

        Args:
            file_path (str): The file path to load the YAML configuration from.

        Returns:
            Scenario: An instance of the Scenario class.
        """
        with open(file_path, 'r') as yaml_file:
            parameters = yaml.safe_load(yaml_file)
        return cls(**parameters)

    def run(self, runner_function: Any):
        """
        Execute a given function with the scenario parameters.

        Args:
            runner_function (callable): A function to execute with the scenario parameters.

        Returns:
            Any: The result of the runner function.
        """
        if not callable(runner_function):
            raise ValueError("runner_function must be callable.")
        return runner_function(**self.parameters)

class Paths:
    """
    A class to manage important paths for a general Python module.

    Attributes:
        base_dir (str): The base directory for the module.
        data_dir (str): The directory for storing data.
        results_dir (str): The directory for storing results.
    """

    def __init__(self):
        """
        Initialize the Paths object.

        Args:
            base_dir (str): The base directory for the module.
        """
        self.base_dir = pathlib.Path(__file__).parent.parent
        self.data_dir = pathlib.Path(self.base_dir / "data")
        self.results_dir = pathlib.Path(self.base_dir / "results")
        self.model_dir = pathlib.Path(self.base_dir / "models")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def get_scenario_results_path(self, scenario_name: str) -> str:
        """
        Get the path for storing results of a specific scenario.

        Args:
            scenario_name (str): The name of the scenario.

        Returns:
            str: The path to the scenario's results directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_dir = os.path.join(self.results_dir, f"{scenario_name}_{timestamp}")
        os.makedirs(scenario_dir, exist_ok=True)
        return scenario_dir

# Example usage
if __name__ == "__main__":
    # Define paths
    paths = Paths()

    # Define multiple scenarios
    scenarios = [
        Scenario(name="Scenario 1", duration=24, battery_capacity=100, fcr_allocation=50),
        Scenario(name="Scenario 2", duration=48, battery_capacity=200, fcr_allocation=75),
        Scenario(name="Scenario 3", duration=72, battery_capacity=150, fcr_allocation=60)
    ]

    # Define a sample runner function
    def sample_runner(name, duration, battery_capacity, fcr_allocation, **kwargs):
        print(f"Running scenario '{name}' with:")
        print(f"Duration: {duration} hours")
        print(f"Battery Capacity: {battery_capacity} kWh")
        print(f"FCR Allocation: {fcr_allocation} MW")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        return "Run completed."

    # Run all scenarios and save their configurations
    for scenario in scenarios:
        result_dir = paths.get_scenario_results_path(scenario.parameters['name'])
        scenario_yaml_path = os.path.join(result_dir, "config.yaml")
        scenario.to_yaml(scenario_yaml_path)
        result = scenario.run(sample_runner)
        print(result)
