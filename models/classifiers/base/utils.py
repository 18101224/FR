import itertools
from typing import List, Optional, Tuple, Union
import safetensors
import torch
from torch import Tensor
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


import shutil
import re


def get_all_folders(root, basename_string=None, sort=False):
    all_folders = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        all_folders += [os.path.join(dirpath, dir) for dir in dirnames]
    if sort:
        all_folders = natural_sort(all_folders)
    if basename_string is None:
        return all_folders
    all_folders = list(filter(lambda x: basename_string in os.path.basename(x), all_folders))
    return all_folders


def get_all_files(root, extension_list=['.csv'], sort=False):

    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        all_files += [os.path.join(dirpath, file) for file in filenames]
    if sort:
        all_files = natural_sort(all_files)
    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    return all_files


def copy_project_files(code_dir, save_path):

    print('copying files from {}'.format(code_dir))
    print('copying files to {}'.format(save_path))
    py_files = get_all_files(code_dir, extension_list=['.py'])
    os.makedirs(save_path, exist_ok=True)
    for py_file in py_files:
        os.makedirs(os.path.dirname(py_file.replace(code_dir, save_path)), exist_ok=True)
        shutil.copy(py_file, py_file.replace(code_dir, save_path))

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]  # Convert key to string here
    return sorted(l, key=alphanum_key)


def make_basename(path, basename_depth=1, ignore_ext=True):
    if ignore_ext:
        _path = os.path.splitext(path)[0]
    else:
        _path = path
    if basename_depth == 1:
        return os.path.basename(_path)
    else:
        return "/".join(_path.split('/')[-basename_depth:])



def match_basenames(path_list1, path_list2, ignore_ext=True):
    result = {}
    if ignore_ext:
        name_to_path1 = {os.path.splitext(os.path.basename(path))[0]: path for path in path_list1}
        name_to_path2 = {os.path.splitext(os.path.basename(path))[0]: path for path in path_list2}
    else:
        name_to_path1 = {os.path.basename(path): path for path in path_list1}
        name_to_path2 = {os.path.basename(path): path for path in path_list2}
    for name in name_to_path1.keys():
        if name in name_to_path2.keys():
            result[name] = (name_to_path1[name], name_to_path2[name])
    return result

def match_basenames_multiple(list_of_list, column_names, basename_depth=1, ignore_ext=True):
    result = {}
    name_to_paths = []
    assert len(list_of_list) == len(column_names)
    for lst in list_of_list:
        name_to_path = {make_basename(path, basename_depth=basename_depth, ignore_ext=ignore_ext): path for path in lst}
        name_to_paths.append(name_to_path)
    common_keys = set.intersection(*map(set, name_to_paths))
    result = {}
    for key in common_keys:
        result[key] = {col_name:name_to_path[key] for col_name, name_to_path in zip(column_names, name_to_paths)}
    return result


def replace_extension(filename, new_extension):
    """Replace the current extension of a filename with a new extension.

    Args:
    - filename (str): The original filename.
    - new_extension (str): The new extension to replace with.

    Returns:
    - str: Filename with the new extension.
    """
    # Strip the starting '.' from new_extension if it's present
    new_extension = new_extension.lstrip('.')

    # Get the base name of the file (without extension)
    base_name = os.path.splitext(filename)[0]

    return f"{base_name}.{new_extension}"

def get_parameter_device(parameter: torch.nn.Module):
    parameters_and_buffers = itertools.chain(parameter.parameters(), parameter.buffers())
    first_tensor = next(parameters_and_buffers, None)
    if first_tensor is not None:
        return first_tensor.device

    def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    first_tuple = next(gen)
    return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module):
    params = tuple(parameter.parameters())
    if len(params) > 0:
        return params[0].dtype

    buffers = tuple(parameter.buffers())
    if len(buffers) > 0:
        return buffers[0].dtype

    def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    first_tuple = next(gen)
    return first_tuple[1].dtype


def get_parent_directory(save_path: Union[str, os.PathLike]) -> Path:
    path_obj = Path(save_path)
    return path_obj.parent

def get_base_name(save_path: Union[str, os.PathLike]) -> str:
    path_obj = Path(save_path)
    return path_obj.name

def load_state_dict_from_path(path: Union[str, os.PathLike]):
    # Load a state dict from a path.
    if 'safetensors' in path:
        state_dict = safetensors.torch.load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu")
    return state_dict

def replace_extension(path, new_extension):
    if not new_extension.startswith('.'):
        new_extension = '.' + new_extension
    return os.path.splitext(path)[0] + new_extension

def make_config_path(save_path):
    config_path = replace_extension(save_path, '.yaml')
    return config_path

def save_config(config, config_path):
    assert isinstance(config, dict) or isinstance(config, DictConfig)
    os.makedirs(get_parent_directory(config_path), exist_ok=True)
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    OmegaConf.save(config, config_path)


def save_state_dict_and_config(state_dict, config, save_path):
    os.makedirs(get_parent_directory(save_path), exist_ok=True)

    # save config dict
    config_path = make_config_path(save_path)
    save_config(config, config_path)

    # Save the model
    if 'safetensors' in save_path:
        safetensors.torch.save_file(state_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(state_dict, save_path)
