import os
from typing import List, Dict
import json
import threading

import carb
import numpy as np
import h5py
import pandas as pd
import torch
from collections import defaultdict

import omni.kit
import omni.usd
import omni.replicator.core as rep
import asyncio 

from omni.syntheticdata.scripts.SyntheticData import SyntheticData

from omni.replicator.core import functional as F
from omni.replicator.core import AnnotatorRegistry
from omni.replicator.core import BackendDispatch
from omni.replicator.core.backends import BackendGroup, BaseBackend
from omni.replicator.core.utils import skeleton_data_utils
from omni.replicator.core import Writer
from omni.replicator.core.writers_default.tools import colorize_distance, colorize_normals

from .io_functions import write_dataframe_hdf5




# Helpers for writing hdf5 file from rl_games dataframe 


def parse_column_structure(df_columns):
    """
    Parse DataFrame column names to determine the nested group structure of the hdf5 file 
    
    return:
    - structure: dict mapping main groups to their subgroups and columns
    """
    structure = defaultdict(list)
    
    for col in df_columns:
        if '/' in col:
            # column has subgroup structure: "main_group/subgroup/column"

            # ie. obs/right_eef_pos creates one level of nestin g
            # obs/datagen_info/eef_pose/left creates 3 levels of nesting 

            parts = col.split('/')
            main_group = parts[0]
            subgroup_path = '/'.join(parts[1:])
            structure[main_group].append(subgroup_path)
        else:
            # root column goes directly under "demo group" 
            structure['root'].append(col)
    
    return dict(structure)


# pretty print structure for debugging 
def print_structure(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}/")
            print_structure(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value.shape}")

# take in the parsed column structure from above and create nested datasets in hdf5 file 
# tree structure -- every group is either a dataset in its immediate group or a subgroup 
def create_nested_datasets(demo_group, df, structure):
    """
    Create nested datasets in HDF5 based on the parsed structure.
    """
    for main_group, subgroups in structure.items():
        if main_group == 'root':
            # root groups 
            for col_name in subgroups:
                data_series = df[col_name]
                if isinstance(data_series.iloc[0], torch.Tensor):
                    stacked_data = torch.stack(data_series.tolist()).numpy()
                else:
                    stacked_data = np.stack(data_series.values)
                demo_group.create_dataset(col_name, data=stacked_data)
        else:
            # subgroups 
            group_obj = demo_group.create_group(main_group)
            
            # Organize columns by their immediate subgroup
            subgroup_dict = defaultdict(list)
            for col_path in subgroups:
                parts = col_path.split('/')
                if len(parts) == 1:
                    # direct dataset 
                    subgroup_dict['root'].append((parts[0], col_path))
                else:
                    # nested subgroup
                    subgroup_dict[parts[0]].append(('/'.join(parts[1:]), col_path))
            
            # Create datasets and nested subgroups
            for immediate_subgroup, column_info in subgroup_dict.items():
                if immediate_subgroup == 'root':
                    # Create datasets directly in the main group
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

def dataframe_to_nested_hdf5(df, hdf5_path, demo_name="demo_0", env_args: Dict | None = None):
    """
    convert dataframe with nested column naming to HDF5 file with nested group structure 
    
    @param
    - df: DataFrame with columns like "obs/right_eef_pos", "actions", "states/articulation" etc.
    - hdf5_path: Path to save the HDF5 file
    - demo_name: Name for this demo subgroup
    """
    
    # entry function for dataframe -> hdf5 demo conversion 
    structure = parse_column_structure(df.columns)
    
    
    with h5py.File(hdf5_path, 'w') as f:
        dataset_group = f.create_group('data')
        if env_args is not None:
            dataset_group.attrs['env_args'] = json.dumps(env_args)
        demo_group = dataset_group.create_group(demo_name)
        # also mirror env_args at file root for easier discovery
        if env_args is not None:
            f.attrs['env_args'] = json.dumps(env_args)
        
        create_nested_datasets(demo_group, df, structure)

    # todo: incorporate replicator backend 

def add_demo_with_nested_structure(df, hdf5_path, demo_name, env_args: Dict | None = None):
    """
    Add a new demo with nested structure to an existing HDF5 file.
    """
    
    structure = parse_column_structure(df.columns)
    
    with h5py.File(hdf5_path, 'a') as f:
        # prefer 'data' group, fall back to existing 'dataset' if present, else create 'data'
        if 'data' in f:
            dataset_group = f['data']
        elif 'dataset' in f:
            dataset_group = f['dataset']
        else:
            dataset_group = f.create_group('data')
        if env_args is not None:
            # set or update env_args on the chosen top-level group
            dataset_group.attrs['env_args'] = json.dumps(env_args)
        if demo_name in dataset_group:
            #print(f"[WARNING] Demo {demo_name} already exists in the HDF5 file. Overwriting...")
            raise ValueError(f"Demo {demo_name} already exists in the HDF5 file.")
            del dataset_group[demo_name]
        

        demo_group = dataset_group.create_group(demo_name)
        create_nested_datasets(demo_group, df, structure)

def read_nested_demo(hdf5_path, demo_name="demo_0"):
    """
    Read a demo with nested structure from HDF5 file.
    
    Returns a nested dictionary of PyTorch tensors.
    """
    def h5_to_dict(group):
        """Recursively convert HDF5 group to dictionary."""
        data_dict = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                data_dict[key] = h5_to_dict(item)
            else:
                data_dict[key] = torch.from_numpy(item[:])
        return data_dict
    
    with h5py.File(hdf5_path, 'r') as f:
        demo_group = f[f'data/{demo_name}']
        return h5_to_dict(demo_group)

# Example usage:


def dataframe_to_hdf5(df, hdf5_path, obs_column='obs', actions_column='actions', 
                     initial_state_column='initial_state', states_column='states'):
    """
    Convert a DataFrame of trajectory data to a nested HDF5 file.
    
    Parameters:
    - df: DataFrame containing trajectory data
    - hdf5_path: Path to save the HDF5 file
    - obs_column: Column name for observations
    - actions_column: Column name for actions  
    - initial_state_column: Column name for initial states
    - states_column: Column name for states
    """
    
    with h5py.File(hdf5_path, 'w') as f:
        dataset_group = f.create_group('dataset')
        
        
        demo_group = dataset_group.create_group(demo_name)
        
        # Create obs group and dataset from the series
        obs_group = demo_group.create_group('obs')
        obs_series = df[obs_column]

__version__ = '0.0.2'
class AsyncWriter(Writer):
    """async writer taken from basic writer implementation at https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/writers_default/basicwriter.py?ref_type=heads

    

    Args:
        output_dir:
            Output directory string that indicates the directory to save the results.
        s3_bucket:
            The S3 Bucket name to write to. If not provided, disk backend will be used instead. Default: ``None``.
            This backend requires that AWS credentials are set up in ``~/.aws/credentials``.
            See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration
        s3_region:
            If provided, this is the region the S3 bucket will be set to. Default: ``us-east-1``
        s3_endpoint:
            If provided, this endpoint URL will be used instead of the default.
        semantic_types:
            List of semantic types to consider when filtering annotator data. Default: ``["class"]``
        rgb:
            Boolean value that indicates whether the ``rgb``/``LdrColor`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        bounding_box_2d_tight:
            Boolean value that indicates whether the ``bounding_box_2d_tight`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        bounding_box_2d_loose:
            Boolean value that indicates whether the ``bounding_box_2d_loose`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        semantic_segmentation:
            Boolean value that indicates whether the ``semantic_segmentation`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        instance_id_segmentation:
            Boolean value that indicates whether the ``instance_id_segmentation`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        instance_segmentation:
            Boolean value that indicates whether the ``instance_segmentation`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        distance_to_camera:
            Boolean value that indicates whether the ``distance_to_camera`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        distance_to_image_plane:
            Boolean value that indicates whether the ``distance_to_image_plane`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        bounding_box_3d:
            Boolean value that indicates whether the ``bounding_box_3d`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        occlusion:
            Boolean value that indicates whether the ``occlusion`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        normals:
            Boolean value that indicates whether the ``normals`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        motion_vectors:
            Boolean value that indicates whether the ``motion_vectors`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        camera_params:
            Boolean value that indicates whether the ``camera_params`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        pointcloud:
            Boolean value that indicates whether the ``pointcloud`` annotator will be activated
            and the data will be written or not. Default: ``False``.
        pointcloud_include_unlabelled:
            If ``True``, pointcloud annotator will capture any prim in the camera's perspective, not matter if it has
            semantics or not. If ``False``, only prims with semantics will be captured.
            Defaults to ``False``.
        image_output_format:
            String that indicates the format of saved RGB images. Default: ``"png"``
        colorize_semantic_segmentation:
            If ``True``, semantic segmentation is converted to an image where semantic IDs are mapped to colors
            and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a ``uint32`` PNG image.
            Defaults to ``True``.
        colorize_instance_id_segmentation:
            If ``True``, instance id segmentation is converted to an image where instance IDs are mapped to colors.
            and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a ``uint32`` PNG image.
            Defaults to ``True``.
        colorize_instance_segmentation:
            If ``True``, instance segmentation is converted to an image where instance are mapped to colors.
            and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a ``uint32`` PNG image.
            Defaults to ``True``.
        colorize_depth:
            If ``True``, will output an additional PNG image for depth for visualization
            Defaults to ``False``.
        frame_padding:
            Pad the frame number with leading zeroes.  Default: ``4``
        semantic_filter_predicate:
            A string specifying a semantic filter predicate as a disjunctive normal form of semantic type, labels.

            Examples :
                "typeA : labelA & !labelB | labelC , typeB: labelA ; typeC: labelD"
                "typeA : * ; * : labelA"
        use_common_output_dir:
            If ``True``, output for each annotator coming from multiple render products are saved under a common directory
            with the render product as the filename prefix (eg. <render_product_name>_<annotator_name>_<sequence>.<format>).
            If ``False``, multiple render product outputs are placed into their own directory
            (eg. <render_product_name>/<annotator_name>_<sequence>.<format>). Setting is ignored if using the writer with
            a single render product. Defaults to ``False``.
        backend: Optionally pass a backend to use. If specified, `output_dir` and `s3_<>` arguments may be omitted. If
            both are provided, the backends will be grouped.


    Example:
        >>> import omni.replicator.core as rep
        >>> import carb
        >>> camera = rep.create.camera()
        >>> render_product = rep.create.render_product(camera, (1024, 1024))
        >>> writer = rep.WriterRegistry.get("BasicWriter")
        >>> tmp_dir = carb.tokens.get_tokens_interface().resolve("${temp}/rgb")
        >>> writer.initialize(output_dir=tmp_dir, rgb=True)
        >>> writer.attach([render_product])
        >>> rep.orchestrator.run()
    """

    def __init__(
        self,
        output_dir: str = None,
        s3_bucket: str = None,
        s3_region: str = None,
        s3_endpoint: str = None,
        semantic_types: List[str] = None,
        rgb: bool = False,
        bounding_box_2d_tight: bool = False,
        bounding_box_2d_loose: bool = False,
        semantic_segmentation: bool = False,
        instance_id_segmentation: bool = False,
        instance_segmentation: bool = False,
        distance_to_camera: bool = False,
        distance_to_image_plane: bool = False,
        bounding_box_3d: bool = False,
        occlusion: bool = False,
        normals: bool = False,
        motion_vectors: bool = False,
        camera_params: bool = False,
        pointcloud: bool = False,
        pointcloud_include_unlabelled: bool = False,
        image_output_format: str = "png",
        colorize_semantic_segmentation: bool = True,
        colorize_instance_id_segmentation: bool = True,
        colorize_instance_segmentation: bool = True,
        colorize_depth: bool = False,
        skeleton_data: bool = False,
        frame_padding: int = 4,
        semantic_filter_predicate: str = None,
        use_common_output_dir: bool = False,
        backend: BaseBackend = None,
    ):
        self._output_dir = output_dir
        self.data_structure = "annotator"
        self.use_common_output_dir = use_common_output_dir
        self._backend = None
        if s3_bucket:
            self._backend = BackendDispatch(
                key_prefix=output_dir,
                bucket=s3_bucket,
                region=s3_region,
                endpoint_url=s3_endpoint,
            )
        elif output_dir:
            self._backend = BackendDispatch(output_dir=output_dir)

        if backend and self._backend:
            self._backend = BackendGroup([backend, *self._backend._backends])
        elif backend:
            self._backend = backend

        if not self._backend:
            raise ValueError("No `backend`, `output_dir` or `s3_` parameter specified, unable to initialize writer.")

        self.backend = self._backend
        self._frame_id = 0
        self._sequence_id = 0
        self._image_output_format = image_output_format
        self._output_data_format = {}
        self.annotators = []
        self.version = __version__
        self._frame_padding = frame_padding

        self.colorize_semantic_segmentation = colorize_semantic_segmentation
        self.colorize_instance_id_segmentation = colorize_instance_id_segmentation
        self.colorize_instance_segmentation = colorize_instance_segmentation
        self.colorize_depth = colorize_depth

        self.num_demos_written = 0
        
        # persistent HDF5 file handles keyed by absolute path
        self._file_map: Dict[str, h5py.File] = {}
        self._file_lock = threading.Lock()
        # env args cache
        if not hasattr(self, '_env_args'):
            self._env_args = None

        is_default_semantic_filter = semantic_filter_predicate is None
        # Specify the semantic types that will be included in output
        if semantic_types is not None:
            if semantic_filter_predicate is None:
                semantic_filter_predicate = ":*; ".join(semantic_types) + ":*"
            else:
                raise ValueError(
                    "`semantic_types` and `semantic_filter_predicate` are mutually exclusive. Please choose only one."
                )
        elif is_default_semantic_filter:
            semantic_filter_predicate = "class:*"

        # Set the global semantic filter predicate
        # FIXME: don't set the global semantic filter predicate after support of multiple instances of annotators
        if semantic_filter_predicate is not None:
            SyntheticData.Get().set_instance_mapping_semantic_filter(semantic_filter_predicate)

        # RGB
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))

        # Bounding Box 2D
        if bounding_box_2d_tight:
            if is_default_semantic_filter:
                self.annotators.append("bounding_box_2d_tight_fast")
            else:
                self.annotators.append(
                    AnnotatorRegistry.get_annotator(
                        "bounding_box_2d_tight_fast",
                        init_params={
                            "semanticFilter": semantic_filter_predicate,
                        },
                    )
                )

        if bounding_box_2d_loose:
            if is_default_semantic_filter:
                self.annotators.append("bounding_box_2d_loose_fast")
            else:
                self.annotators.append(
                    AnnotatorRegistry.get_annotator(
                        "bounding_box_2d_loose_fast",
                        init_params={
                            "semanticFilter": semantic_filter_predicate,
                        },
                    )
                )

        # Semantic Segmentation
        if semantic_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "semantic_segmentation",
                    init_params={
                        "colorize": colorize_semantic_segmentation,
                        "semanticFilter": semantic_filter_predicate,
                    },
                )
            )

        # Instance Segmentation
        if instance_id_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "instance_id_segmentation_fast", init_params={"colorize": colorize_instance_id_segmentation}
                )
            )

        # Instance Segmentation
        if instance_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "instance_segmentation_fast",
                    init_params={
                        "colorize": colorize_instance_segmentation,
                        "semanticFilter": semantic_filter_predicate,
                    },
                )
            )

        # Depth
        if distance_to_camera:
            self.annotators.append(AnnotatorRegistry.get_annotator("distance_to_camera"))

        if distance_to_image_plane:
            self.annotators.append(AnnotatorRegistry.get_annotator("distance_to_image_plane"))

        # Bounding Box 3D
        if bounding_box_3d:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "bounding_box_3d_fast",
                    init_params={
                        "semanticFilter": semantic_filter_predicate,
                    },
                )
            )

        # Motion Vectors
        if motion_vectors:
            self.annotators.append(AnnotatorRegistry.get_annotator("motion_vectors"))

        # Occlusion
        if occlusion:
            self.annotators.append(AnnotatorRegistry.get_annotator("occlusion"))

        # Normals
        if normals:
            self.annotators.append(AnnotatorRegistry.get_annotator("normals"))

        # Camera Params
        if camera_params:
            self.annotators.append(AnnotatorRegistry.get_annotator("camera_params"))

        # Pointcloud
        if pointcloud:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "pointcloud", init_params={"includeUnlabelled": pointcloud_include_unlabelled}
                )
            )

        # Skeleton Data
        if skeleton_data:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("skeleton_data", init_params={"useSkelJoints": False})
            )

        backend_type = "S3" if s3_bucket else "Disk"


    def _write_trajectory_data_hdf5(self, data : pd.DataFrame, out_file : str, debug = False):
        filepath = os.path.join(self._output_dir, out_file)
        
        def _get_or_create_file_handle(filepath: str) -> h5py.File:
            """Get or create persistent file handle for the given filepath."""
            with self._file_lock:
                f = self._file_map.get(filepath)
                if f is None:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    f = h5py.File(filepath, 'a')
                    self._file_map[filepath] = f
                return f
        
        demo_name = f"demo_{self.num_demos_written}"
        env_args = getattr(self, '_env_args', None)
        
        # Use backend-compatible write function with persistent file handle getter
        write_dataframe_hdf5(
            path=filepath,
            data=data,
            backend_instance=self._backend,
            demo_name=demo_name,
            env_args=env_args,
            file_handle_getter=_get_or_create_file_handle,
        )
 
        self.num_demos_written += 1
         

        if debug: 
            loaded_data = read_nested_demo(filepath, demo_name=f"demo_{self.num_demos_written-1}")
            print_structure(loaded_data)




    # for now dont actually use this 
    def write(self, data : pd.DataFrame): 
        return self._write_trajectory_data_hdf5(data, f"trajectory_data.hdf5", debug=True)

    async def write_trajectory_data_async(
        self,
        data: pd.DataFrame,
        out_file: str = "trajectory_data.hdf5",
        debug: bool = False,
    ):

        await asyncio.to_thread(self._write_trajectory_data_hdf5, data, out_file, debug)

    # --- Env args support (to mirror HDF5DatasetFileHandler) ---
    def set_env_args(self, env_args: Dict):
        if not hasattr(self, '_env_args') or self._env_args is None:
            self._env_args = {}
        self._env_args.update(env_args)

    def close(self):
        # flush and close all open files
        with self._file_lock:
            for fp, f in list(self._file_map.items()):
                try:
                    f.flush()
                except Exception:
                    pass
                try:
                    f.close()
                except Exception:
                    pass
                self._file_map.pop(fp, None)

    async def _write(self, data: dict):
        """Write function called from the OgnWriter node on every frame to process annotator output.

        Args:
            data: A dictionary containing the annotator data for the current frame.
        """
        # Check for on_time triggers
        # For each on_time trigger, prefix the output frame number with the trigger counts
        sequence_id = ""
        for trigger_name, call_count in data["trigger_outputs"].items():
            if "on_time" in trigger_name:
                sequence_id = f"{call_count}_{sequence_id}"
        if sequence_id != self._sequence_id:
            self._frame_id = 0
            self._sequence_id = sequence_id

        for annotator_name, annotator_data in data["annotators"].items():
            # Shorten fast annotator names
            if annotator_name.endswith("_fast"):
                annotator_name = annotator_name[:-5]

            is_multi_rp = len(annotator_data) > 1
            for render_product_name, anno_rp_data in annotator_data.items():
                if is_multi_rp:
                    if self.use_common_output_dir:
                        output_path = (
                            os.path.join(annotator_name, render_product_name) + "_"
                        )  # Add render product as prefix
                    else:
                        output_path = (
                            os.path.join(render_product_name, annotator_name) + os.path.sep
                        )  # Legacy behaviour
                else:
                    output_path = ""

                if annotator_name == "rgb" or annotator_name.startswith("Aug"):
                    self._write_rgb(anno_rp_data, output_path)

                elif annotator_name == "normals":
                    self._write_normals(anno_rp_data, output_path)

                elif annotator_name == "distance_to_camera":
                    self._write_distance_to_camera(anno_rp_data, output_path)

                elif annotator_name == "distance_to_image_plane":
                    self._write_distance_to_image_plane(anno_rp_data, output_path)

                elif annotator_name.startswith("semantic_segmentation"):
                    self._write_semantic_segmentation(anno_rp_data, output_path)

                elif annotator_name.startswith("instance_id_segmentation"):
                    self._write_instance_id_segmentation(anno_rp_data, output_path)

                elif annotator_name.startswith("instance_segmentation"):
                    self._write_instance_segmentation(anno_rp_data, output_path)

                elif annotator_name.startswith("motion_vectors"):
                    self._write_motion_vectors(anno_rp_data, output_path)

                elif annotator_name.startswith("occlusion"):
                    self._write_occlusion(anno_rp_data, output_path)

                elif annotator_name.startswith("bounding_box_3d"):
                    self._write_bounding_box_data(anno_rp_data, "3d", output_path)

                elif annotator_name.startswith("bounding_box_2d_loose"):
                    self._write_bounding_box_data(anno_rp_data, "2d_loose", output_path)

                elif annotator_name.startswith("bounding_box_2d_tight"):
                    self._write_bounding_box_data(anno_rp_data, "2d_tight", output_path)

                elif annotator_name.startswith("camera_params"):
                    self._write_camera_params(anno_rp_data, output_path)

                elif annotator_name.startswith("pointcloud"):
                    self._write_pointcloud(anno_rp_data, output_path)

                elif annotator_name.startswith("skeleton_data"):
                    self._write_skeleton(anno_rp_data, output_path)

                elif annotator_name not in ["camera", "resolution"]:
                    carb.log_warn(f"Unknown {annotator_name=}")

        self._frame_id += 1

    def _write_rgb(self, anno_rp_data: dict, output_path: str):
        file_path = (
            f"{output_path}rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.{self._image_output_format}"
        )
        self._backend.schedule(F.write_image, data=anno_rp_data["data"], path=file_path)

    def _write_normals(self, anno_rp_data: dict, output_path: str):
        normals_data = anno_rp_data["data"]
        file_path = f"{output_path}normals_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        colorized_normals_data = colorize_normals(normals_data)
        self._backend.schedule(F.write_image, data=colorized_normals_data, path=file_path)

    def _write_distance_to_camera(self, anno_rp_data: dict, output_path: str):
        dist_to_cam_data = anno_rp_data["data"]
        file_path = f"{output_path}distance_to_camera_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=dist_to_cam_data, path=file_path)
        if self.colorize_depth:
            file_path = (
                f"{output_path}distance_to_camera_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
            )
            self._backend.schedule(
                F.write_image, data=colorize_distance(dist_to_cam_data, near=None, far=None), path=file_path
            )


    
    

    def _write_distance_to_image_plane(self, anno_rp_data: dict, output_path: str):
        dis_to_img_plane_data = anno_rp_data["data"]
        file_path = (
            f"{output_path}distance_to_image_plane_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.schedule(F.write_np, data=dis_to_img_plane_data, path=file_path)
        if self.colorize_depth:
            file_path = (
                f"{output_path}distance_to_image_plane_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
            )
            self._backend.schedule(
                F.write_image, data=colorize_distance(dis_to_img_plane_data, near=None, far=None), path=file_path
            )

    def _write_semantic_segmentation(self, anno_rp_data: dict, output_path: str):
        semantic_seg_data = anno_rp_data["data"]
        height, width = semantic_seg_data.shape[:2]

        file_path = f"{output_path}semantic_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        if self.colorize_semantic_segmentation:
            semantic_seg_data = semantic_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.schedule(F.write_image, data=semantic_seg_data, path=file_path)
        else:
            semantic_seg_data = semantic_seg_data.view(np.uint32).reshape(height, width)
            self._backend.schedule(F.write_image, data=semantic_seg_data, path=file_path)

        id_to_labels = anno_rp_data["idToLabels"]
        file_path = (
            f"{output_path}semantic_segmentation_labels_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        )

        self._backend.schedule(F.write_json, data={str(k): v for k, v in id_to_labels.items()}, path=file_path)

    def _write_instance_id_segmentation(self, anno_rp_data: dict, output_path: str):
        instance_seg_data = anno_rp_data["data"]
        height, width = instance_seg_data.shape[:2]

        file_path = (
            f"{output_path}instance_id_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        )
        if self.colorize_instance_id_segmentation:
            instance_seg_data = instance_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.schedule(F.write_image, data=instance_seg_data, path=file_path)
        else:
            instance_seg_data = instance_seg_data.view(np.uint32).reshape(height, width)
            self._backend.schedule(F.write_image, data=instance_seg_data, path=file_path)

        id_to_labels = anno_rp_data["idToLabels"]
        file_path = f"{output_path}instance_id_segmentation_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        self._backend.schedule(F.write_json, data={str(k): v for k, v in id_to_labels.items()}, path=file_path)

    def _write_instance_segmentation(self, anno_rp_data: dict, output_path: str):
        instance_seg_data = anno_rp_data["data"]
        height, width = instance_seg_data.shape[:2]

        file_path = f"{output_path}instance_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        if self.colorize_instance_segmentation:
            instance_seg_data = instance_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.schedule(F.write_image, data=instance_seg_data, path=file_path)
        else:
            instance_seg_data = instance_seg_data.view(np.uint32).reshape(height, width)
            self._backend.schedule(F.write_image, data=instance_seg_data, path=file_path)

        id_to_labels = anno_rp_data["idToLabels"]
        file_path = f"{output_path}instance_segmentation_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        self._backend.schedule(F.write_json, data={str(k): v for k, v in id_to_labels.items()}, path=file_path)

        id_to_semantics = anno_rp_data["idToSemantics"]
        file_path = f"{output_path}instance_segmentation_semantics_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        self._backend.schedule(F.write_json, data={str(k): v for k, v in id_to_semantics.items()}, path=file_path)

    def _write_motion_vectors(self, anno_rp_data: dict, output_path: str):
        motion_vec_data = anno_rp_data["data"]
        file_path = f"{output_path}motion_vectors_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=motion_vec_data, path=file_path)

    def _write_occlusion(self, anno_rp_data: dict, output_path: str):
        occlusion_data = anno_rp_data["data"]

        file_path = f"{output_path}occlusion_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=occlusion_data, path=file_path)

    def _write_bounding_box_data(self, anno_rp_data: dict, bbox_type: str, output_path: str):
        bbox_data = anno_rp_data["data"]
        id_to_labels = anno_rp_data["idToLabels"]
        prim_paths = anno_rp_data["primPaths"]

        file_path = (
            f"{output_path}bounding_box_{bbox_type}_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.schedule(F.write_np, data=bbox_data, path=file_path)

        labels_file_path = f"{output_path}bounding_box_{bbox_type}_labels_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        self._backend.schedule(F.write_json, data=id_to_labels, path=labels_file_path)

        labels_file_path = f"{output_path}bounding_box_{bbox_type}_prim_paths_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        self._backend.schedule(F.write_json, data=prim_paths, path=labels_file_path)

    def _write_camera_params(self, anno_rp_data: dict, output_path: str):
        camera_data = anno_rp_data
        serializable_data = {}

        for key, val in camera_data.items():
            if isinstance(val, np.ndarray):
                serializable_data[key] = val.tolist()
            else:
                serializable_data[key] = val

        file_path = f"{output_path}camera_params_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        self._backend.schedule(F.write_json, data=serializable_data, path=file_path)

    def _write_pointcloud(self, anno_rp_data: dict, output_path: str):
        pointcloud_data = anno_rp_data["data"]
        pointcloud_rgb = anno_rp_data["pointRgb"].reshape(-1, 4)
        pointcloud_normals = anno_rp_data["pointNormals"].reshape(-1, 4)
        pointcloud_semantic = anno_rp_data["pointSemantic"]
        pointcloud_instance = anno_rp_data["pointInstance"]

        file_path = f"{output_path}pointcloud_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=pointcloud_data, path=file_path)

        rgb_file_path = f"{output_path}pointcloud_rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=pointcloud_rgb, path=rgb_file_path)

        normals_file_path = (
            f"{output_path}pointcloud_normals_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.schedule(F.write_np, data=pointcloud_normals, path=normals_file_path)

        semantic_file_path = (
            f"{output_path}pointcloud_semantic_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.schedule(F.write_np, data=pointcloud_semantic, path=semantic_file_path)

        instance_file_path = (
            f"{output_path}pointcloud_instance_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.schedule(F.write_np, data=pointcloud_instance, path=instance_file_path)

    def _write_skeleton(self, anno_rp_data: dict, output_path: str):
        # "skeletonData" is deprecated
        # skeleton = json.loads(anno_rp_data["skeletonData"])

        skeleton_dict = {}

        skel_name = anno_rp_data["skelName"]
        skel_path = anno_rp_data["skelPath"]
        asset_path = anno_rp_data["assetPath"]
        animation_variant = anno_rp_data["animationVariant"]
        skeleton_parents = skeleton_data_utils.get_skeleton_parents(
            anno_rp_data["numSkeletons"], anno_rp_data["skeletonParents"], anno_rp_data["skeletonParentsSizes"]
        )
        rest_global_translations = skeleton_data_utils.get_rest_global_translations(
            anno_rp_data["numSkeletons"],
            anno_rp_data["restGlobalTranslations"],
            anno_rp_data["restGlobalTranslationsSizes"],
        )
        rest_local_translations = skeleton_data_utils.get_rest_local_translations(
            anno_rp_data["numSkeletons"],
            anno_rp_data["restLocalTranslations"],
            anno_rp_data["restLocalTranslationsSizes"],
        )
        rest_local_rotations = skeleton_data_utils.get_rest_local_rotations(
            anno_rp_data["numSkeletons"],
            anno_rp_data["restLocalRotations"],
            anno_rp_data["restLocalRotationsSizes"],
        )
        global_translations = skeleton_data_utils.get_global_translations(
            anno_rp_data["numSkeletons"],
            anno_rp_data["globalTranslations"],
            anno_rp_data["globalTranslationsSizes"],
        )
        local_rotations = skeleton_data_utils.get_local_rotations(
            anno_rp_data["numSkeletons"], anno_rp_data["localRotations"], anno_rp_data["localRotationsSizes"]
        )
        translations_2d = skeleton_data_utils.get_translations_2d(
            anno_rp_data["numSkeletons"], anno_rp_data["translations2d"], anno_rp_data["translations2dSizes"]
        )
        skeleton_joints = skeleton_data_utils.get_skeleton_joints(anno_rp_data["skeletonJoints"])
        joint_occlusions = skeleton_data_utils.get_joint_occlusions(
            anno_rp_data["numSkeletons"], anno_rp_data["jointOcclusions"], anno_rp_data["jointOcclusionsSizes"]
        )
        occlusion_types = skeleton_data_utils.get_occlusion_types(
            anno_rp_data["numSkeletons"], anno_rp_data["occlusionTypes"], anno_rp_data["occlusionTypesSizes"]
        )
        in_view = anno_rp_data["inView"]

        for skel_num in range(anno_rp_data["numSkeletons"]):
            skeleton_dict[f"skeleton_{skel_num}"] = {}
            skeleton_dict[f"skeleton_{skel_num}"]["skel_name"] = skel_name[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["skel_path"] = skel_path[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["asset_path"] = asset_path[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["animation_variant"] = animation_variant[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["skeleton_parents"] = (
                skeleton_parents[skel_num].tolist() if skeleton_parents else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["rest_global_translations"] = (
                rest_global_translations[skel_num].tolist() if rest_global_translations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["rest_local_translations"] = (
                rest_local_translations[skel_num].tolist() if rest_local_translations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["rest_local_rotations"] = (
                rest_local_rotations[skel_num].tolist() if rest_local_rotations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["global_translations"] = (
                global_translations[skel_num].tolist() if global_translations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["local_rotations"] = (
                local_rotations[skel_num].tolist() if local_rotations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["translations_2d"] = (
                translations_2d[skel_num].tolist() if translations_2d else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["skeleton_joints"] = (
                skeleton_joints[skel_num] if skeleton_joints else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["joint_occlusions"] = (
                joint_occlusions[skel_num].tolist() if joint_occlusions else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["occlusion_types"] = (
                occlusion_types[skel_num] if occlusion_types else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["in_view"] = bool(in_view[skel_num]) if in_view.any() else False

        file_path = f"{output_path}skeleton_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"

        self._backend.schedule(F.write_json, data=skeleton_dict, path=file_path)


