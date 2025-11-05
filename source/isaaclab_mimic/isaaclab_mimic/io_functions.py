# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r'''
The io_functions.py module provides a set of utility functions, which can be easily utilized to manage I/O operations
within different backends. Additional functions can be registered with this module to seamlessly expand the capabilities
of backends to handle a variety of I/O tasks.

Example:

.. code:: python

    def my_io_function(backend_instance, **kwargs)
        """My IO function

        Parameters:
            backend_instance: An instance of the backend derived from BaseBackend and registered with BackendRegistry, used
                for executing the I/O operation. For instance, in the statement
                ``backend_instance.schedule(io_functions.my_io_function, kwargs)``, backend_instance is the instance of the
                backend used to schedule the my_function operation with specified keyword arguments.
            **kwargs: Keyword arguments that can be employed to provide supplementary information or customization for the
                I/O operation. These arguments can be specified for I/O functions that require specific configurations, such
                as ``write_jpeg``.
        """
'''

import io
import json
import os
import pickle
import platform
import threading
from contextlib import nullcontext
from typing import Union, Dict, Callable, Optional

import numpy as np
import warp as wp
import h5py
import pandas as pd
import torch
from collections import defaultdict
from omni.replicator.core.bindings._omni_replicator_exrwriter import load_exr_from_stream, save_exr_to_stream
from PIL import Image

from omni.replicator.core.backends import BaseBackend, DiskBackend


def _to_pil_image(data):
    if isinstance(data, wp.array):
        data = data.numpy()

    if isinstance(data, np.ndarray):
        if data.shape[-1] > 3 and len(data.shape) == 3:
            data = Image.fromarray(data, "RGBA")
        elif data.shape[-1] == 3 and len(data.shape) == 3:
            data = Image.fromarray(data, "RGB")
        elif data.shape[-1] == 1 and len(data.shape) == 3:
            data = Image.fromarray(data[:, :, 0], "L")
        else:
            if data.dtype == np.uint16:
                data = Image.fromarray(data, "I;16")
            else:
                data = Image.fromarray(data)

    if not isinstance(data, Image.Image):
        raise ValueError(f"Expected image data to be a numpy ndarray, warp array or PIL.Image, got {type(data)}")

    return data


def write_image(
    path: str, data: Union[np.ndarray, wp.array, Image.Image], backend_instance: BaseBackend = DiskBackend, **kwargs
) -> None:
    """
    Write image data to a specified path.
    Supported image extensions include: [jpeg, jpg, png, exr]

    Args:
        path: Write path URI
        data: Image data
        backend_instance: Backend to use to write. Defaults to ``DiskBackend``.
        kwargs: Specify additional save parameters, typically specific to the image file type.
    """
    if isinstance(data, wp.array):
        data = data.numpy()

    ext = os.path.splitext(path)[-1][1:]
    if ext.lower() not in ["jpeg", "jpg", "png", "exr"]:
        raise ValueError(f"Could not write image to path `{path}`, image extension `{ext}` is not supported.")

    if ext.lower() in ["jpeg", "jpg", "png"]:
        data = _to_pil_image(data)

        if ext.lower() in ["jpeg", "jpg"]:
            data = data.convert("RGB")
            write_jpeg(path, data, backend_instance=backend_instance, **kwargs)
        else:
            write_png(path, data, backend_instance=backend_instance, **kwargs)

    elif ext.lower() == "exr":
        write_exr(path, data, backend_instance=backend_instance, **kwargs)


def write_jpeg(
    path: str,
    data: Union[np.ndarray, wp.array],
    backend_instance: BaseBackend = DiskBackend,
    quality: int = 75,
    progressive: bool = False,
    optimize: bool = False,
    **kwargs,
) -> None:
    """
    Write image data to JPEG.

    Args:
        path: Write path URI
        data: Data to write
        backend_instance: Backend to use to write. Defaults to ``DiskBackend``. Defaults to ``DiskBackend``.
        quality: The image quality, on a scale from 0 (worst) to 95 (best), or the string keep. The default is 75.
            Values above 95 should be avoided; 100 disables portions of the JPEG compression algorithm, and results in
            large files with hardly any gain in image quality. The value keep is only valid for JPEG files and will
            retain the original image quality level, subsampling, and qtables.
        progressive: Indicates that this image should be stored as a progressive JPEG file.
        optimize: Reduce file size, may be slower. Indicates that the encoder should make an extra pass over the image
            in order to select optimal encoder settings.
        kwargs: Additional parameters may be specified and can be found within the PILLOW documentation:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg-saving
    """
    data = _to_pil_image(data)
    buf = io.BytesIO()
    data.save(buf, format="jpeg", quality=quality, optimize=optimize, progressive=progressive, **kwargs)
    backend_instance.write_blob(path, buf.getvalue())


def write_png(
    path: str,
    data: Union[np.ndarray, wp.array],
    backend_instance: BaseBackend = DiskBackend,
    compress_level: int = 3,
    **kwargs,
) -> None:
    """
    Write image data to PNG.


    Args:
        path: Write path URI
        data: Data to write
        backend_instance: Backend to use to write. Defaults to ``DiskBackend``.
        compress_level: Specifies ZLIB compression level. Compression is specified as a value between [0, 9] where 1 is
            fastest and 9 provides the best compression. A value of 0 provides no compression. Defaults to 3.
        **kwargs: Additional parameters may be specified and can be found within the PILLOW documentation:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#png-saving
    """
    data = _to_pil_image(data)
    buf = io.BytesIO()
    data.save(buf, format="png", compress_level=compress_level, **kwargs)
    backend_instance.write_blob(path, buf.getvalue())


def _write_exr_imageio(
    path: str, data: Union[np.ndarray, wp.array], backend_instance: BaseBackend = DiskBackend, exr_flag=None, **kwargs
) -> None:
    """
    Write data to EXR.

    Args:
        path: Write path URI
        data: Data to write
        backend_instance: Backend to use to write. Defaults to ``DiskBackend``.
        exr_flag from FIF_EXR:
            - imageio.plugins.freeimage.IO_FLAGS.EXR_DEFAULT: Save data as half with piz-based wavelet compression
            - imageio.plugins.freeimage.IO_FLAGS.EXR_FLOAT: Save data as float instead of as half (not recommended)
            - imageio.plugins.freeimage.IO_FLAGS.EXR_NONE: Save with no compression
            - imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP: Save with zlib compression, in blocks of 16 scan lines
            - imageio.plugins.freeimage.IO_FLAGS.EXR_PIZ: Save with piz-based wavelet compression
            - imageio.plugins.freeimage.IO_FLAGS.EXR_PXR24: Save with lossy 24-bit float compression
            - imageio.plugins.freeimage.IO_FLAGS.EXR_B44: Save with lossy 44% float compression - goes to 22% when
              combined with EXR_LC
            - imageio.plugins.freeimage.IO_FLAGS.EXR_LC: Save images with one luminance and two chroma channels, rather
              than as RGB (lossy compression)
    """
    import imageio

    if isinstance(data, wp.array):
        data = data.numpy()

    # Download freeimage dll, will only download once if not present
    # from https://imageio.readthedocs.io/en/v2.8.0/format_exr-fi.html#exr-fi
    imageio.plugins.freeimage.download()
    if exr_flag is None and platform.machine() != "aarch64":
        # Flag for x86_64, not supported on ARM at the moment, tracked in OMPE-46846
        exr_flag = imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP

        exr_bytes = imageio.imwrite(
            imageio.RETURN_BYTES,
            data,
            format="exr",
            flags=exr_flag,
        )
    else:
        exr_bytes = imageio.imwrite(
            imageio.RETURN_BYTES,
            data,
            format="exr",
        )
    backend_instance.write_blob(path, exr_bytes)


def write_exr(
    path: str,
    data: Union[np.ndarray, wp.array],
    backend_instance: BaseBackend = DiskBackend,
    half_precision: bool = False,
    **kwargs,
) -> None:
    """
    Write data to EXR.

    Args:
        path: Write path URI
        data: Data to write
        backend_instance: Backend to use to write. Defaults to ``DiskBackend``.
        half_precision: bool, optional
            Save data as half precision instead of full precision. Default to False.
        **kwargs: If "exr_flag" is provided, legacy imageio implementation is used.
    """
    if "exr_flag" in kwargs:
        return _write_exr_imageio(path, data, backend_instance, kwargs["exr_flag"])

    if isinstance(data, wp.array):
        data = data.numpy()

    buf = io.BytesIO()
    save_exr_to_stream(buf, data, half_precision)
    backend_instance.write_blob(path, buf.getvalue())


def write_json(
    path,
    data,
    backend_instance=None,
    encoding="utf-8",
    errors="strict",
    **kwargs,
) -> None:
    """
    Write json data to a specified path.

    Args:
        path: Write path URI
        data: Data to write
        backend_instance: Backend to use to write. Defaults to ``DiskBackend``.
        encoding: This parameter specifies the encoding to be used. For a list of all encoding schemes, please visit:
            https://docs.python.org/3/library/codecs.html#standard-encodings
        errors: This parameter specifies an error handling scheme when encoding the json string data. The default for
            errors is 'strict' which means that the encoding errors raise a UnicodeError. Other possible values are
            'ignore', 'replace', 'xmlcharrefreplace', 'backslashreplace' and any othername registered via
            codecs.register_error().
        **kwargs: Additional JSON encoding parameters may be supplied. See
            https://docs.python.org/3/library/json.html#json.dump for full list.
    """

    buf = io.BytesIO()
    buf.write(
        json.dumps(
            data,
            **kwargs,
        ).encode(encoding, errors=errors)
    )
    backend_instance.write_blob(path, buf.getvalue())


def write_pickle(
    path: str, data: Union[np.ndarray, wp.array], backend_instance: BaseBackend = DiskBackend, **kwargs
) -> None:
    """
    Write pickle data to a specified path.

    Args:
        path: Write path URI
        data: Data to write
        backend_instance: Backend to use to write. Defaults to ``DiskBackend``.
        **kwargs: Additional Pickle encoding parameters may be supplied. See
            https://docs.python.org/3/library/pickle.html#pickle.Pickler for full list.
    """
    buf = io.BytesIO()
    pickle.dump(data, buf, **kwargs)
    backend_instance.write_blob(path, buf.getvalue())


def write_np(
    path: str,
    data: Union[np.ndarray, wp.array],
    backend_instance: BaseBackend = DiskBackend,
    allow_pickle: bool = True,
    fix_imports: bool = True,
) -> None:
    """
    Write numpy data to a specified path.
    Save parameters are detailed here: https://numpy.org/doc/stable/reference/generated/numpy.save.html

    Args:
        path: Write path URI
        data: Data to write
        backend_instance: Backend to use to write. Defaults to ``DiskBackend``.
        allow_pickle : bool, optional
            Allow saving object arrays using Python pickles. Reasons for disallowing
            pickles include security (loading pickled data can execute arbitrary
            code) and portability (pickled objects may not be loadable on different
            Python installations, for example if the stored objects require libraries
            that are not available, and not all pickled data is compatible between
            Python 2 and Python 3).
            Default to True.
        fix_imports : bool, optional
            Only useful in forcing objects in object arrays on Python 3 to be
            pickled in a Python 2 compatible way. If ``fix_imports`` is True, pickle
            will try to map the new Python 3 names to the old module names used in
            Python 2, so that the pickle data stream is readable with Python 2. Defaults
            to True
    """
    if isinstance(data, wp.array):
        data = data.numpy()

    buf = io.BytesIO()
    np.save(buf, data, allow_pickle=allow_pickle, fix_imports=fix_imports)
    backend_instance.write_blob(path, buf.getvalue())


def _parse_column_structure(df_columns):
    """Parse DataFrame column names to determine the nested group structure of the hdf5 file."""
    structure = defaultdict(list)
    for col in df_columns:
        if '/' in col:
            parts = col.split('/')
            main_group = parts[0]
            subgroup_path = '/'.join(parts[1:])
            structure[main_group].append(subgroup_path)
        else:
            structure['root'].append(col)
    return dict(structure)


def _create_nested_datasets(demo_group, df, structure):
    """Create nested datasets in HDF5 based on the parsed structure."""
    for main_group, subgroups in structure.items():
        if main_group == 'root':
            for col_name in subgroups:
                data_series = df[col_name]
                if isinstance(data_series.iloc[0], torch.Tensor):
                    stacked_data = torch.stack(data_series.tolist()).numpy()
                else:
                    stacked_data = np.stack(data_series.values)
                demo_group.create_dataset(col_name, data=stacked_data)
        else:
            group_obj = demo_group.create_group(main_group)
            subgroup_dict = defaultdict(list)
            for col_path in subgroups:
                parts = col_path.split('/')
                if len(parts) == 1:
                    subgroup_dict['root'].append((parts[0], col_path))
                else:
                    subgroup_dict[parts[0]].append(('/'.join(parts[1:]), col_path))
            
            for immediate_subgroup, column_info in subgroup_dict.items():
                if immediate_subgroup == 'root':
                    for dataset_name, col_path in column_info:
                        data_series = df[f"{main_group}/{col_path}"]
                        if isinstance(data_series.iloc[0], torch.Tensor):
                            stacked_data = torch.stack(data_series.tolist()).numpy()
                        else:
                            stacked_data = np.stack(data_series.values)
                        group_obj.create_dataset(dataset_name, data=stacked_data)
                else:
                    subgroup_obj = group_obj.create_group(immediate_subgroup)
                    for nested_path, col_path in column_info:
                        data_series = df[f"{main_group}/{col_path}"]
                        if isinstance(data_series.iloc[0], torch.Tensor):
                            stacked_data = torch.stack(data_series.tolist()).numpy()
                        else:
                            stacked_data = np.stack(data_series.values)
                        subgroup_obj.create_dataset(nested_path, data=stacked_data)


def write_dataframe_hdf5(
    path: str,
    data: pd.DataFrame,
    backend_instance: BaseBackend = DiskBackend,
    demo_name: str = "demo_0",
    env_args: Optional[Dict] = None,
    file_handle_getter: Optional[Callable[[str], h5py.File]] = None,
    file_lock: Optional[threading.Lock] = None,
    **kwargs,
) -> None:
    """
    Write DataFrame data to HDF5 file with nested structure.
    
    Args:
        path: Write path URI for the HDF5 file
        data: DataFrame containing trajectory data with nested column naming (e.g., "obs/right_eef_pos")
        backend_instance: Backend to use for directory creation and path management. Defaults to ``DiskBackend``.
        demo_name: Name for this demo subgroup. Defaults to "demo_0".
        env_args: Optional dictionary of environment arguments to store as attributes.
        file_handle_getter: Optional callable that takes a filepath and returns an open h5py.File handle.
            If provided, this handle will be used for writing. If None, a new file will be opened.
        **kwargs: Additional parameters (unused, for compatibility with backend pattern).
    """
    lock_ctx = file_lock if file_lock is not None else nullcontext()

    with lock_ctx:
        structure = _parse_column_structure(data.columns)

        if file_handle_getter is not None:
            # Use provided file handle (for persistent handles)
            f = file_handle_getter(path)
        else:
            # Ensure directory exists when possible
            dir_path = os.path.dirname(path)
            if hasattr(backend_instance, "make_dirs"):
                if dir_path:
                    backend_instance.make_dirs(dir_path)
            elif dir_path:
                os.makedirs(dir_path, exist_ok=True)
            f = h5py.File(path, 'a')

        group = f['data'] if 'data' in f else f.create_group('data')
        if env_args is not None:
            try:
                group.attrs['env_args'] = json.dumps(env_args)
                f.attrs['env_args'] = json.dumps(env_args)
            except Exception:
                pass

        if demo_name in group:
            if file_handle_getter is None:
                f.close()
            raise ValueError(f"Demo {demo_name} already exists in the HDF5 file.")

        demo_group = group.create_group(demo_name)
        _create_nested_datasets(demo_group, data, structure)

        try:
            f.flush()
        except Exception:
            pass

        if file_handle_getter is None:
            f.close()


def read_exr(
    path: str,
    backend_instance: BaseBackend = DiskBackend,
) -> np.ndarray:
    """Read an EXR image and return it as a NumPy ``ndarray``.

    Args:
        path (str): Path to the EXR file to read.
        backend_instance (BaseBackend, optional): Backend to use when reading the
            file. If an *instance* of a backend is supplied, its
            :py:meth:`read_blob` method is used and the image is decoded from
            memory.  If a backend *class* (e.g. ``DiskBackend``) is given, the
            path is read directly from disk. Defaults to ``DiskBackend``.

    Returns:
        numpy.ndarray: The decoded image data. The array shape is
        ``(H, W)`` for single-channel images or ``(H, W, C)`` for multi-channel
        images. The dtype matches the source file (typically ``float32`` or
        ``float16``).
    """
    exr_bytes = backend_instance.read_blob(path)
    buf = io.BytesIO(exr_bytes)
    return load_exr_from_stream(buf)
