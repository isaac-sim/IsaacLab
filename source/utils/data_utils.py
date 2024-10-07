import glob
from typing import Dict, Callable

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_numpy(v) for v in x]
    else:
        return x


def readOBJ(file):
    V, Vt, F, Ft = [], [], [], []
    with open(file, 'r') as f:
        T = f.readlines()
    for t in T:
        # 3D vertex
        if t.startswith('v '):
            v = [float(n) for n in t.replace('v ', '').split(' ')]
            V += [v]
        # UV vertex
        elif t.startswith('vt '):
            v = [float(n) for n in t.replace('vt ', '').split(' ')]
            Vt += [v]
        # Face
        elif t.startswith('f '):
            idx = [n.split('/') for n in t.replace('f ', '').split(' ')]
            f = [int(n[0]) - 1 for n in idx]
            F += [f]
    V = np.array(V, np.float32)
    Vt = np.array(Vt, np.float32)
    if Ft:
        assert len(F) == len(
            Ft), 'Inconsistent .obj file, mesh and UV map do not have the same number of faces'
    else:
        Vt, Ft = None, None
    F = np.array(F, np.int32)
    return V, F, Vt, Ft


def read_h5_dict(path, data_names=None, exclude_names=[]):
    hf = h5py.File(path, 'r')
    data = {}
    hf_names = hf.keys()
    data_names = hf_names if data_names is None else data_names
    for name in data_names:
        if name in exclude_names or name not in hf_names:
            continue
        d = np.array(hf.get(name))
        data[name] = d
    hf.close()
    return data


def store_h5_dict(path, data_dict, compress=True):
    hf = h5py.File(path, 'w')
    for k, v in data_dict.items():
        if compress:
            hf.create_dataset(k, data=v, compression="lzf")
        else:
            hf.create_dataset(k, data=v)
    hf.close()


def compute_stats(data_list, prefix='', mode="simple"):
    if mode == "simple":
        out = {'{}_num'.format(prefix): len(data_list),
               '{}_mean'.format(prefix): np.mean(data_list),
               '{}_std'.format(prefix): np.std(data_list),
               '{}_50th_pct'.format(prefix): np.percentile(data_list, 50)}
    else:
        out = {'{}_num'.format(prefix): len(data_list),
               '{}_mean'.format(prefix): np.mean(data_list),
               '{}_std'.format(prefix): np.std(data_list),
               '{}_min'.format(prefix): np.min(data_list),
               '{}_max'.format(prefix): np.max(data_list),
               '{}_10th_pct'.format(prefix): np.percentile(data_list, 10),
               '{}_50th_pct'.format(prefix): np.percentile(data_list, 50),
               '{}_90th_pct'.format(prefix): np.percentile(data_list, 90)}
    return out


def update_config(default_cfg, extra_params):
    for k, v in extra_params.items():
        OmegaConf.update(default_cfg, k, v)
    return default_cfg


def find_best_checkpoint(model_path, epoch=None):
    if 'ckpt' not in model_path:
        ckpt_pattern = f'{model_path}/*.ckpt'
    else:
        ckpt_pattern = model_path
    all_ckpts = glob.glob(ckpt_pattern)
    best_ckpt = sorted(all_ckpts)[-1]
    print('Found checkpoint', best_ckpt)
    return best_ckpt




def aggre_dicts(dicts, stack=False):
    out = {}
    for k in dicts[0].keys():
        out[k] = [d[k] for d in dicts]
        if stack:
            out[k] = np.stack(out[k], axis=0)
    return out


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class SmartDict(Dict):
    """
    A slicable dictionary for numpy and tensor object.
    d = {"a": torch.ones(8, 3), "b": torch.ones(8, 4)}
    custom_dict = SlicedDict(d)
    sliced_result = custom_dict[:3]
    sliced_result = custom_dict[:3, 1]
    sliced_result = custom_dict[0, 0]
    sliced_result = custom_dict[:3, 1:2]
    """

    @property
    def device(self):
        for v in self.values():
            if torch.is_tensor(v):
                return v.device
        return None

    @property
    def dtype(self):
        for v in self.values():
            if torch.is_tensor(v):
                return v.dtype
        return None

    def __len__(self):
        for v in self.values():
            return len(v)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            super().__setitem__(key, value)
        elif isinstance(key, slice) or isinstance(key, tuple):
            assert len(value) == len(self), "The length of the value should be the same as the length of the dict"
            for k, v in self.items():
                v[key] = value[k]

    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key,
                                                tuple) or key is None or isinstance(key,
                                                                                    int):
            sliced_dict = SmartDict()
            # check if the values are of the same length
            if key is not None:
                lens = [len(v) for v in self.values()]
                assert len(
                    set(lens)) == 1, "Try to slice a SmartDict with varied length of values."

            for k, v in self.items():
                sliced_dict[k] = v[key]
            return sliced_dict
        else:
            return super().__getitem__(key)

    def to_numpy(self):
        new_dict = SmartDict()
        for k, v in self.items():
            if torch.is_tensor(v):
                new_dict[k] = v.detach().float().cpu().numpy()
            elif isinstance(v, SmartDict):
                new_dict[k] = v.to_numpy()
            else:
                new_dict[k] = v
        return new_dict

    def to_tensor(self, device="cuda"):
        new_dict = SmartDict()
        for k, v in self.items():
            if isinstance(v, np.ndarray):
                new_dict[k] = torch.tensor(v, device=device)
            elif torch.is_tensor(v):
                new_dict[k] = v.to(device)
            elif isinstance(v, Dict):
                new_dict[k] = v.to_tensor(device)
            else:
                new_dict[k] = v
        return new_dict


def np_concat_list_dict(list_dict):
    for k, v in list_dict.items():
        if len(v) > 0:
            list_dict[k] = np.concatenate(v, axis=0)
    return list_dict


def stack_dict(dicts, axis=0):
    out = SmartDict()
    for k in dicts[0].keys():
        stackable = False
        if torch.is_tensor(dicts[0][k]):
            stack = torch.stack
            stackable = True
        elif isinstance(dicts[0][k], np.ndarray):
            stack = np.stack
            stackable = True
        if stackable:
            out[k] = stack([d[k] for d in dicts], axis)
        else:
            out[k] = [d[k] for d in dicts]
    return out


def cat_dict(dicts, axis=0):
    out = SmartDict()
    for k in dicts[0].keys():
        catable = False
        if torch.is_tensor(dicts[0][k]):
            cat = torch.cat
            catable = True
        elif isinstance(dicts[0][k], np.ndarray):
            cat = np.concatenate
            catable = True
        if catable:
            out[k] = cat([d[k] for d in dicts], axis)
        else:
            out[k] = [d[k] for d in dicts]
    return out


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = SmartDict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def dict_list_to_list_dict(dict_list):
    list_dict = {}
    for k in dict_list[0].keys():
        list_dict[k] = []
    for d in dict_list:
        for k, v in d.items():
            list_dict[k].append(v)
    return list_dict


def torch_stack_list_dict(list_dict):
    for k, v in list_dict.items():
        if len(v) > 0:
            list_dict[k] = torch.stack(v, dim=0)
    return list_dict


def interp1d(x, y, new_x):
    # x: N, sorted in ascending order, torch.tensor
    # y: N x D, values at x, torch.tensor
    # new_x:  M , are the new x values to interpolate at, torch.tensor
    # Return the linear interpolation of y at new_x
    # M x D
    assert x.shape[0] == y.shape[0]
    assert len(x.shape) == 1
    assert len(new_x.shape) == 1
    assert x.device == y.device
    assert x.device == new_x.device

    # find the indices of the new_x values in x
    # M
    idx = torch.searchsorted(x, new_x) - 1
    # M
    idx = torch.clamp(idx, 0, x.shape[0] - 1)
    # M
    idx_next = torch.clamp(idx + 1, 0, x.shape[0] - 1)

    interpolation_weights = (new_x - x[idx]) / (x[idx_next] - x[idx])
    interpolation_weights = interpolation_weights.unsqueeze(1)

    # M x D
    y_interp = y[idx] + (y[idx_next] - y[idx]) * interpolation_weights
    return y_interp
