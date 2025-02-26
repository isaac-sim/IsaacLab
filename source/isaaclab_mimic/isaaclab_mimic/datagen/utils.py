# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ipywidgets import widgets
from IPython.display import display
import os
from isaaclab.utils.datasets import HDF5DatasetFileHandler

def get_nested_value(d, keys):
    """Retrieve a nested value from dictionary d using list of keys."""
    for k in keys:
        d = d[k]
    return d

def update_nested_value(d, keys, value):
    """Update a nested value in dictionary d using list of keys."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

def get_parameter_input(param_name, current_val, allowed_range, update_fn, env=None):
    """Get parameter input using ipywidgets with immediate value updates."""
    
    if isinstance(current_val, (tuple, list)):
        # Get step size from allowed_range if provided, else use default
        step_size = allowed_range[2] if len(allowed_range) > 2 else 0.01
        
        # Create two text input widgets for tuple values
        text1 = widgets.FloatText(
            value=current_val[0],
            description=f"{param_name}[0]",
            style={'description_width': 'initial'},
            placeholder=f"Default: {current_val[0]}",
            step=step_size
        )
        text2 = widgets.FloatText(
            value=current_val[1],
            description=f"{param_name}[1]",
            style={'description_width': 'initial'},
            placeholder=f"Default: {current_val[1]}",
            step=step_size
        )
        
        # Add value changed callbacks
        def on_value1_change(change):
            if change['new'] < allowed_range[0] or change['new'] > allowed_range[1]:
                print(f"Value {change['new']} is out of allowed range {allowed_range[:2]}. Keeping current value.")
                text1.value = change['old']
            elif change['new'] > text2.value:
                print(f"Minimum value {change['new']} must be less than maximum value {text2.value}. Keeping current value.")
                text1.value = change['old']
            else:
                new_tuple = (change['new'], text2.value)
                update_fn(new_tuple)
                if env is not None:
                    env.reset()

        def on_value2_change(change):
            if change['new'] < allowed_range[0] or change['new'] > allowed_range[1]:
                print(f"Value {change['new']} is out of allowed range {allowed_range[:2]}. Keeping current value.")
                text2.value = change['old']
            elif change['new'] < text1.value:
                print(f"Maximum value {change['new']} must be greater than minimum value {text1.value}. Keeping current value.")
                text2.value = change['old']
            else:
                new_tuple = (text1.value, change['new'])
                update_fn(new_tuple)
                if env is not None:
                    env.reset()
        
        text1.observe(on_value1_change, names='value')
        text2.observe(on_value2_change, names='value')
        
        # Create help text showing the allowed range
        help_text = widgets.HTML(
            value=f'<p style="color:gray">Allowed range: {allowed_range[:2]}</p>'
        )
        
        # Display widgets
        display(text1)
        display(text2)
        display(help_text)
        
        return (text1, text2)
    else:
        # Create single text input widget for float values
        step_size = allowed_range[2] if len(allowed_range) > 2 else 0.01
        text = widgets.FloatText(
            value=current_val,
            description=param_name,
            style={'description_width': 'initial'},
            placeholder=f"Default: {current_val}",
            step=step_size 
        )
        
        def on_value_change(change):
            if change['new'] < allowed_range[0] or change['new'] > allowed_range[1]:
                print(f"Value {change['new']} is out of allowed range {allowed_range[:2]}. Keeping current value.")
                text.value = change['old']
            else:
                update_fn(change['new'])
                if env is not None:
                    env.reset()
                    
        text.observe(on_value_change, names='value')
        
        # Create help text showing the allowed range
        help_text = widgets.HTML(
            value=f'<p style="color:gray">Allowed range: {allowed_range[:2]}</p>'
        )
        
        # Display widgets
        display(text)
        display(help_text)
        
        return text

def interactive_update_randomizable_params(event_term, param_config, param_path="", env=None):
    """Interactive parameter updates using ipywidgets."""
    inputs = []
    print("event_term:", event_term, "param_config:", param_config, "param_path:", param_path)
    
    for key, allowed_range in param_config.items():
        print("key:", key, "allowed_range:", allowed_range)
        current_path = f"{param_path}.{key}" if param_path else key
        keys = current_path.split(".")
        
        if isinstance(allowed_range, dict):
            interactive_update_randomizable_params(event_term, allowed_range, current_path, env)
        else:
            try:
                current_val = get_nested_value(event_term.params, keys)
                
                def make_update_fn(k):
                    def update_fn(new_val):
                        update_nested_value(event_term.params, k, new_val)
                        print(f"Updated '{current_path}' to {new_val}.")
                    return update_fn
                
                input_widget = get_parameter_input(
                    current_path, 
                    current_val, 
                    allowed_range,
                    make_update_fn(keys),
                    env=env
                )
                inputs.append((keys, input_widget))
            except KeyError:
                print(f"Key '{current_path}' not found in event_term.params; skipping.")
                continue
    
    return inputs

def setup_output_paths(output_file_path: str) -> tuple[str, str]:
    """Set up output directory and get file name for dataset generation.
    
    Args:
        output_file_path: Full path to the desired output file
        
    Returns:
        tuple containing:
            - output_dir: Path to the output directory
            - output_file_name: Name of the output file without extension
    """
    output_dir = os.path.dirname(output_file_path)
    output_file_name = os.path.splitext(os.path.basename(output_file_path))[0]
    
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    return output_dir, output_file_name

def get_env_name_from_dataset(input_file_path: str) -> str:
    """Get environment name from an input dataset file.
    
    Args:
        input_file_path: Path to the input dataset file
        
    Returns:
        env_name: Name of the environment from the dataset
        
    Raises:
        FileNotFoundError: If the input file does not exist
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"The dataset file {input_file_path} does not exist.")
        
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(input_file_path)
    env_name = dataset_file_handler.get_env_name()
    if env_name is None:
        raise ValueError("Environment name not found in dataset")
    
    return env_name
