#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of scenarios

Copyright (c) by Institute for High Voltage Equipment and Grids, 
Digitalization and Energy Economics (IAEW), RWTH Aachen University, 
28.01.2025, s.kortmann. All rights reserved.
"""
import yaml
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

# Example usage
if __name__ == "__main__":
    # Define a sample scenario
    scenario = Scenario(name="Test Scenario",
                        duration=24,
                        battery_capacity=100,
                        fcr_allocation=50)

    # Update parameters
    scenario.update_parameters(duration=48, efficiency=0.9)

    # Export to YAML
    scenario.to_yaml("scenario_config.yaml")

    # Load from YAML
    new_scenario = Scenario.from_yaml("scenario_config.yaml")

    # Define a sample runner function
    def sample_runner(name, duration, battery_capacity, fcr_allocation, **kwargs):
        print(f"Running scenario '{name}' with:")
        print(f"Duration: {duration} hours")
        print(f"Battery Capacity: {battery_capacity} kWh")
        print(f"FCR Allocation: {fcr_allocation} MW")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        return "Run completed."

    # Run the scenario
    result = new_scenario.run(sample_runner)
    print(result)

    # Define multiple scenarios
    scenarios = [
        Scenario(name="Scenario 1", duration=24, battery_capacity=100, fcr_allocation=50),
        Scenario(name="Scenario 2", duration=48, battery_capacity=200, fcr_allocation=75),
        Scenario(name="Scenario 3", duration=72, battery_capacity=150, fcr_allocation=60)
    ]


    # Run all scenarios
    for scenario in scenarios:
        result = scenario.run(sample_runner)
        print(result)