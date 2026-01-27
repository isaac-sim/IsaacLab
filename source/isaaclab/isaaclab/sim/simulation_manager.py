# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import weakref
from collections import OrderedDict

import carb
import isaacsim.core.utils.numpy as np_utils
import isaacsim.core.utils.torch as torch_utils
import isaacsim.core.utils.warp as warp_utils
import omni.kit
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.stage import get_current_stage, get_current_stage_id
from pxr import PhysxSchema

from .isaac_events import IsaacEvents


class SimulationManager:
    """This class provide functions that take care of many time-related events such as
    warm starting simulation in order for the physics data to be retrievable.
    Adding/ removing callback functions that gets triggered with certain events such as a physics step,
    on post reset, on physics ready..etc."""

    _warmup_needed = True
    #_timeline = omni.timeline.get_timeline_interface()
    _message_bus = carb.eventdispatcher.get_eventdispatcher()
    _physx_sim_interface = omni.physx.get_physx_simulation_interface()
    _physx_interface = omni.physx.get_physx_interface()
    _physics_sim_view = None
    _physics_sim_view__warp = None
    _backend = "torch"
    _carb_settings = carb.settings.get_settings()
    _physics_scene_apis = OrderedDict()
    #_callbacks = dict()
    _simulation_manager_interface = None
    _simulation_view_created = False
    #_assets_loaded = True
    #_assets_loading_callback = None
    #_assets_loaded_callback = None
    _default_physics_scene_idx = -1

    # callback handles
    #_warm_start_callback = None
    #_on_stop_callback = None
    #_post_warm_start_callback = None
    #_stage_open_callback = None

    # Add callback state tracking
    #_callbacks_enabled = {
    #    "warm_start": True,
    #    "on_stop": True,
    #    "post_warm_start": True,
    #    "stage_open": True,
    #}

    @classmethod
    def _initialize(cls) -> None:
        # Initialize all callbacks as enabled by default
        SimulationManager.enable_all_default_callbacks(True)
        SimulationManager._track_physics_scenes()

    # ------------------------------------------------------------------------------------------------------------------    
    # TODO: Removing this as the callbacks handling are moved to the SimulationContext class.
    #@classmethod
    #def _setup_warm_start_callback(cls) -> None:
    #    if cls._callbacks_enabled["warm_start"] and cls._warm_start_callback is None:
    #        cls._warm_start_callback = cls._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
    #            int(omni.timeline.TimelineEventType.PLAY), cls._warm_start
    #        )

    #@classmethod
    #def _setup_on_stop_callback(cls) -> None:
    #    if cls._callbacks_enabled["on_stop"] and cls._on_stop_callback is None:
    #        cls._on_stop_callback = cls._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
    #            int(omni.timeline.TimelineEventType.STOP), cls._on_stop
    #        )

    #@classmethod
    #def _setup_post_warm_start_callback(cls) -> None:
    #    if cls._callbacks_enabled["post_warm_start"] and cls._post_warm_start_callback is None:
    #        cls._post_warm_start_callback = cls._message_bus.observe_event(
    #            event_name=IsaacEvents.PHYSICS_WARMUP.value,
    #            on_event=cls._create_simulation_view,
    #            observer_name="SimulationManager._post_warm_start_callback",
    #        )

    #@classmethod
    #def _setup_stage_open_callback(cls) -> None:
    #    if cls._callbacks_enabled["stage_open"] and cls._stage_open_callback is None:
    #        cls._stage_open_callback = cls._message_bus.observe_event(
    #            event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.OPENED),
    #            on_event=cls._post_stage_open,
    #            observer_name="SimulationManager._stage_open_callback",
    #        )

    @classmethod
    def _clear(cls) -> None:
        # Use callback management system for main callbacks
        #cls.enable_all_default_callbacks(False)

        # Handle additional cleanup not covered by enable_all_default_callbacks
        cls._physics_sim_view = None
        cls._physics_sim_view__warp = None
        cls._assets_loading_callback = None
        cls._assets_loaded_callback = None
        cls._simulation_manager_interface.reset()
        cls._physics_scene_apis.clear()
        cls._callbacks.clear()

    def _post_stage_open(event) -> None:
        SimulationManager._simulation_manager_interface.reset()
        SimulationManager._physics_scene_apis.clear()
        SimulationManager._callbacks.clear()
        SimulationManager._track_physics_scenes()
        SimulationManager._assets_loaded = True
        SimulationManager._assets_loading_callback = None
        SimulationManager._assets_loaded_callback = None

        # TODO: This should not be needed as we should be synchronous.
        #def _assets_loading(event):
        #    SimulationManager._assets_loaded = False

        #def _assets_loaded(event):
        #    SimulationManager._assets_loaded = True

        #SimulationManager._assets_loading_callback = SimulationManager._message_bus.observe_event(
        #    event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.ASSETS_LOADING),
        #    on_event=_assets_loading,
        #    observer_name="SimulationManager._assets_loading_callback",
        #)

        #SimulationManager._assets_loaded_callback = SimulationManager._message_bus.observe_event(
        #    event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.ASSETS_LOADED),
        #    on_event=_assets_loaded,
        #    observer_name="SimulationManager._assets_loaded_callback",
        #)

    # ------------------------------------------------------------------------------------------------------------------    
    # TODO: This is very USD centric, we should not need this. Also we should have only one physics scene.
    #def _track_physics_scenes() -> None:
    #    def add_physics_scenes(physics_scene_prim_path):
    #        prim = get_prim_at_path(physics_scene_prim_path)
    #        if prim.GetTypeName() == "PhysicsScene":
    #            SimulationManager._physics_scene_apis[physics_scene_prim_path] = PhysxSchema.PhysxSceneAPI.Apply(prim)

    #    def remove_physics_scenes(physics_scene_prim_path):
    #        # TODO: match physics scene prim path
    #        if physics_scene_prim_path in SimulationManager._physics_scene_apis:
    #            del SimulationManager._physics_scene_apis[physics_scene_prim_path]

    #    SimulationManager._simulation_manager_interface.register_physics_scene_addition_callback(add_physics_scenes)
    #    SimulationManager._simulation_manager_interface.register_deletion_callback(remove_physics_scenes)

    def _warm_start(event) -> None:
        if SimulationManager._carb_settings.get_as_bool("/app/player/playSimulations"):
            SimulationManager.initialize_physics()

    def _on_stop(event) -> None:
        SimulationManager._warmup_needed = True
        if SimulationManager._physics_sim_view:
            SimulationManager._physics_sim_view.invalidate()
            SimulationManager._physics_sim_view = None
            SimulationManager._simulation_view_created = False
        if SimulationManager._physics_sim_view__warp:
            SimulationManager._physics_sim_view__warp.invalidate()
            SimulationManager._physics_sim_view__warp = None

    def _create_simulation_view(event) -> None:
        if "cuda" in SimulationManager.get_physics_sim_device() and SimulationManager._backend == "numpy":
            SimulationManager._backend = "torch"
            carb.log_warn("changing backend from numpy to torch since numpy backend cannot be used with GPU piplines")
        SimulationManager._physics_sim_view = omni.physics.tensors.create_simulation_view(
            SimulationManager.get_backend(), stage_id=get_current_stage_id()
        )
        SimulationManager._physics_sim_view.set_subspace_roots("/")
        #SimulationManager._physics_sim_view__warp = omni.physics.tensors.create_simulation_view(
        #    "warp", stage_id=get_current_stage_id()
        #)
        #SimulationManager._physics_sim_view__warp.set_subspace_roots("/")
        SimulationManager._physx_interface.update_simulation(SimulationManager.get_physics_dt(), 0.0)
        SimulationManager._message_bus.dispatch_event(IsaacEvents.SIMULATION_VIEW_CREATED.value, payload={})
        SimulationManager._simulation_view_created = True
        SimulationManager._message_bus.dispatch_event(IsaacEvents.PHYSICS_READY.value, payload={})

    # ------------------------------------------------------------------------------------------------------------------    
    # TODO: Removing this as we should use our internal tools.
    #@classmethod
    #def _get_backend_utils(cls) -> str:
    #    if SimulationManager._backend == "numpy":
    #        return np_utils
    #    elif SimulationManager._backend == "torch":
    #        return torch_utils
    #    elif SimulationManager._backend == "warp":
    #        return warp_utils
    #    else:
    #        raise Exception(
    #            f"Provided backend is not supported: {SimulationManager.get_backend()}. Supported: torch, numpy, warp."
    #        )

    # ------------------------------------------------------------------------------------------------------------------    
    # TODO: This is very USD centric, we should not need this. Also we should have only one physics scene.
    #@classmethod
    #def _get_physics_scene_api(cls, physics_scene: str = None):
    #    if physics_scene is None:
    #        if len(SimulationManager._physics_scene_apis) > 0:
    #            physics_scene_api = list(SimulationManager._physics_scene_apis.values())[
    #                SimulationManager._default_physics_scene_idx
    #            ]
    #        else:
    #            # carb.log_warn("Physics scene is not found in stage")
    #            return None
    #    else:
    #        if physics_scene in SimulationManager._physics_scene_apis:
    #            physics_scene_api = SimulationManager._physics_scene_apis[physics_scene]
    #        else:
    #            carb.log_warn("physics scene specified {} doesn't exist".format(physics_scene))
    #            return None
    #    return physics_scene_api

    @classmethod
    def set_backend(cls, val: str) -> None:
        SimulationManager._backend = val

    @classmethod
    def get_backend(cls) -> str:
        return SimulationManager._backend

    @classmethod
    def initialize_physics(cls) -> None:
        if SimulationManager._warmup_needed:
            SimulationManager._physx_interface.force_load_physics_from_usd()
            SimulationManager._physx_interface.start_simulation()
            SimulationManager._physx_interface.update_simulation(SimulationManager.get_physics_dt(), 0.0)
            SimulationManager._physx_sim_interface.fetch_results()
            SimulationManager._message_bus.dispatch_event(IsaacEvents.PHYSICS_WARMUP.value, payload={})
            SimulationManager._warmup_needed = False

    # TODO: Should handle simulation time itself.
    @classmethod
    def get_simulation_time(cls):
        #return SimulationManager._simulation_manager_interface.get_simulation_time()
        return self._simulation_time

    # TODO: Should handle simulation time itself.
    @classmethod
    def get_num_physics_steps(cls):
        return SimulationManager._simulation_manager_interface.get_num_physics_steps()

    # TODO: Doesn't need to know if the simulation is simulating or not.
    #@classmethod
    #def is_simulating(cls):
    #    return SimulationManager._simulation_manager_interface.is_simulating()

    # TODO: Doesn't need to know if the simulation is paused or not.
    #@classmethod
    #def is_paused(cls):
    #    return SimulationManager._simulation_manager_interface.is_paused()

    @classmethod
    def get_physics_sim_view(cls):
        return SimulationManager._physics_sim_view

    @classmethod
    def set_default_physics_scene(cls, physics_scene_prim_path: str):
        if SimulationManager._warm_start_callback is None:
            carb.log_warn("Calling set_default_physics_scene while SimulationManager is not tracking physics scenes")
            return
        if physics_scene_prim_path in SimulationManager._physics_scene_apis:
            SimulationManager._default_physics_scene_idx = list(SimulationManager._physics_scene_apis.keys()).index(
                physics_scene_prim_path
            )
        elif is_prim_path_valid(physics_scene_prim_path):
            prim = get_prim_at_path(physics_scene_prim_path)
            if prim.GetTypeName() == "PhysicsScene":
                SimulationManager._physics_scene_apis[physics_scene_prim_path] = PhysxSchema.PhysxSceneAPI.Apply(prim)
                SimulationManager._default_physics_scene_idx = list(SimulationManager._physics_scene_apis.keys()).index(
                    physics_scene_prim_path
                )
        else:
            raise Exception("physics scene specified {} doesn't exist".format(physics_scene_prim_path))

    @classmethod
    def get_default_physics_scene(cls) -> str:
        if len(SimulationManager._physics_scene_apis) > 0:
            return list(SimulationManager._physics_scene_apis.keys())[SimulationManager._default_physics_scene_idx]
        else:
            carb.log_warn("No physics scene is found in stage")
            return None

    @classmethod
    def step(cls, render: bool = False):
        if render:
            raise Exception(
                "Stepping the renderer is not supported at the moment through SimulationManager, use SimulationContext instead."
            )
        SimulationManager._physx_sim_interface.simulate(
            SimulationManager.get_physics_dt(physics_scene=None), SimulationManager.get_simulation_time()
        )
        SimulationManager._physx_sim_interface.fetch_results()

    @classmethod
    def set_physics_sim_device(cls, val) -> None:
        if "cuda" in val:
            parsed_device = val.split(":")
            if len(parsed_device) == 1:
                device_id = SimulationManager._carb_settings.get_as_int("/physics/cudaDevice")
                if device_id < 0:
                    SimulationManager._carb_settings.set_int("/physics/cudaDevice", 0)
                    device_id = 0
            else:
                SimulationManager._carb_settings.set_int("/physics/cudaDevice", int(parsed_device[1]))
            SimulationManager._carb_settings.set_bool("/physics/suppressReadback", True)
            SimulationManager.set_broadphase_type("GPU")
            SimulationManager.enable_gpu_dynamics(flag=True)
            SimulationManager.enable_fabric(enable=True)
        elif "cpu" == val.lower():
            SimulationManager._carb_settings.set_bool("/physics/suppressReadback", False)
            # SimulationManager._carb_settings.set_int("/physics/cudaDevice", -1)
            SimulationManager.set_broadphase_type("MBP")
            SimulationManager.enable_gpu_dynamics(flag=False)
        else:
            raise Exception("Device {} is not supported.".format(val))

    @classmethod
    def get_physics_sim_device(cls) -> str:
        supress_readback = SimulationManager._carb_settings.get_as_bool("/physics/suppressReadback")
        if (not SimulationManager._physics_scene_apis and supress_readback) or (
            supress_readback
            and SimulationManager.get_broadphase_type() == "GPU"
            and SimulationManager.is_gpu_dynamics_enabled()
        ):
            device_id = SimulationManager._carb_settings.get_as_int("/physics/cudaDevice")
            if device_id < 0:
                SimulationManager._carb_settings.set_int("/physics/cudaDevice", 0)
                device_id = 0
            return f"cuda:{device_id}"
        else:
            return "cpu"

    def set_physics_dt(cls, dt: float = 1.0 / 60.0, physics_scene: str = None) -> None:
        """Sets the physics dt on the physics scene provided.

        Args:
            dt (float, optional): physics dt. Defaults to 1.0/60.0.
            physics_scene (str, optional): physics scene prim path. Defaults to first physics scene found in the stage.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
            ValueError: Physics dt must be a >= 0.
            ValueError: Physics dt must be a <= 1.0.
        """
        if physics_scene is None:
            physics_scene_apis = SimulationManager._physics_scene_apis.values()
        else:
            physics_scene_apis = [SimulationManager._get_physics_scene_api(physics_scene=physics_scene)]

        for physics_scene_api in physics_scene_apis:
            if dt < 0:
                raise ValueError("physics dt cannot be <0")
            # if no stage or no change in physics timestep, exit.
            if get_current_stage() is None:
                return
            if dt == 0:
                physics_scene_api.GetTimeStepsPerSecondAttr().Set(0)
            elif dt > 1.0:
                raise ValueError("physics dt must be <= 1.0")
            else:
                steps_per_second = int(1.0 / dt)
                physics_scene_api.GetTimeStepsPerSecondAttr().Set(steps_per_second)
        return

    @classmethod
    def get_physics_dt(cls, physics_scene: str = None) -> str:
        """
         Returns the current physics dt.

        Args:
            physics_scene (str, optional): physics scene prim path.

        Raises:
            Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

        Returns:
            float: physics dt.
        """
        physics_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
        if physics_scene_api is None:
            return 1.0 / 60.0
        physics_hz = physics_scene_api.GetTimeStepsPerSecondAttr().Get()
        if physics_hz == 0:
            return 0.0
        else:
            return 1.0 / physics_hz

    # ------------------------------------------------------------------------------------------------------------------    
    # TODO: Removing this as we will only set the settings on initialization.
    #@classmethod
    #def get_broadphase_type(cls, physics_scene: str = None) -> str:
    #    """Gets current broadcast phase algorithm type.

    #    Args:
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

    #    Returns:
    #        str: Broadcast phase algorithm used.
    #    """
    #    physics_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #    return physics_scene_api.GetBroadphaseTypeAttr().Get()

    #@classmethod
    #def set_broadphase_type(cls, val: str, physics_scene: str = None) -> None:
    #    """Broadcast phase algorithm used in simulation.

    #    Args:
    #        val (str): type of broadcasting to be used, can be "MBP".
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
    #    """
    #    if physics_scene is None:
    #        for path, physx_scene_api in SimulationManager._physics_scene_apis.items():
    #            if not physx_scene_api.GetPrim().IsValid():
    #                continue
    #            if physx_scene_api.GetBroadphaseTypeAttr().Get() is None:
    #                physx_scene_api.CreateBroadphaseTypeAttr(val)
    #            else:
    #                physx_scene_api.GetBroadphaseTypeAttr().Set(val)
    #    else:
    #        physx_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #        if physx_scene_api.GetBroadphaseTypeAttr().Get() is None:
    #            physx_scene_api.CreateBroadphaseTypeAttr(val)
    #        else:
    #            physx_scene_api.GetBroadphaseTypeAttr().Set(val)

    #@classmethod
    #def enable_ccd(cls, flag: bool, physics_scene: str = None) -> None:
    #    """Enables a second broad phase after integration that makes it possible to prevent objects from tunneling
    #       through each other.

    #    Args:
    #        flag (bool): enables or disables ccd on the PhysicsScene
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
    #    """
    #    if physics_scene is None:
    #        for path, physx_scene_api in SimulationManager._physics_scene_apis.items():
    #            if not physx_scene_api.GetPrim().IsValid():
    #                continue
    #            if physx_scene_api.GetEnableCCDAttr().Get() is None:
    #                physx_scene_api.CreateEnableCCDAttr(flag)
    #            else:
    #                physx_scene_api.GetEnableCCDAttr().Set(flag)
    #    else:
    #        physx_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #        if physx_scene_api.GetEnableCCDAttr().Get() is None:
    #            physx_scene_api.CreateEnableCCDAttr(flag)
    #        else:
    #            physx_scene_api.GetEnableCCDAttr().Set(flag)

    #@classmethod
    #def is_ccd_enabled(cls, physics_scene: str = None) -> bool:
    #    """Checks if ccd is enabled.

    #    Args:
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

    #    Returns:
    #        bool: True if ccd is enabled, otherwise False.
    #    """
    #    physx_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #    return physx_scene_api.GetEnableCCDAttr().Get()

    #@classmethod
    #def enable_ccd(cls, flag: bool, physics_scene: str = None) -> None:
    #    """Enables Continuous Collision Detection (CCD).

    #    Args:
    #        flag (bool): enables or disables CCD on the PhysicsScene.
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
    #    """
    #    if flag and "cuda" in SimulationManager.get_physics_sim_device():
    #        carb.log_warn("CCD is not supported on GPU, ignoring request to enable it")
    #        return
    #    if physics_scene is None:
    #        for path, physx_scene_api in SimulationManager._physics_scene_apis.items():
    #            if not physx_scene_api.GetPrim().IsValid():
    #                continue
    #            if physx_scene_api.GetEnableCCDAttr().Get() is None:
    #                physx_scene_api.CreateEnableCCDAttr(flag)
    #            else:
    #                physx_scene_api.GetEnableCCDAttr().Set(flag)
    #    else:
    #        if physics_scene in SimulationManager._physics_scene_apis:
    #            physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
    #            if physx_scene_api.GetEnableCCDAttr().Get() is None:
    #                physx_scene_api.CreateEnableCCDAttr(flag)
    #            else:
    #                physx_scene_api.GetEnableCCDAttr().Set(flag)
    #        else:
    #            raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    #@classmethod
    #def is_ccd_enabled(cls, physics_scene: str = None) -> bool:
    #    """Checks if Continuous Collision Detection (CCD) is enabled.

    #    Args:
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

    #    Returns:
    #        bool: True if CCD is enabled, otherwise False.
    #    """
    #    if physics_scene is None:
    #        if len(SimulationManager._physics_scene_apis) > 0:
    #            physx_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #            return physx_scene_api.GetEnableCCDAttr().Get()
    #        else:
    #            return False
    #    else:
    #        if physics_scene in SimulationManager._physics_scene_apis:
    #            physx_scene_api = SimulationManager._physics_scene_apis[physics_scene]
    #            return physx_scene_api.GetEnableCCDAttr().Get()
    #        else:
    #            raise Exception("physics scene specified {} doesn't exist".format(physics_scene))

    #@classmethod
    #def enable_gpu_dynamics(cls, flag: bool, physics_scene: str = None) -> None:
    #    """Enables gpu dynamics pipeline, required for deformables for instance.

    #    Args:
    #        flag (bool): enables or disables gpu dynamics on the PhysicsScene. If flag is true, CCD is disabled.
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
    #    """
    #    if physics_scene is None:
    #        for path, physx_scene_api in SimulationManager._physics_scene_apis.items():
    #            if not physx_scene_api.GetPrim().IsValid():
    #                continue
    #            if physx_scene_api.GetEnableGPUDynamicsAttr().Get() is None:
    #                physx_scene_api.CreateEnableGPUDynamicsAttr(flag)
    #            else:
    #                physx_scene_api.GetEnableGPUDynamicsAttr().Set(flag)
    #            # Disable CCD for GPU dynamics as its not supported
    #            if flag:
    #                if SimulationManager.is_ccd_enabled():
    #                    carb.log_warn("Disabling CCD for GPU dynamics as its not supported")
    #                    SimulationManager.enable_ccd(flag=False)
    #    else:
    #        physx_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #        if physx_scene_api.GetEnableGPUDynamicsAttr().Get() is None:
    #            physx_scene_api.CreateEnableGPUDynamicsAttr(flag)
    #        else:
    #            physx_scene_api.GetEnableGPUDynamicsAttr().Set(flag)
    #        # Disable CCD for GPU dynamics as its not supported
    #        if flag:
    #            if SimulationManager.is_ccd_enabled(physics_scene=physics_scene):
    #                carb.log_warn("Disabling CCD for GPU dynamics as its not supported")
    #                SimulationManager.enable_ccd(flag=False, physics_scene=physics_scene)
    #        else:
    #            physx_scene_api.GetEnableGPUDynamicsAttr().Set(flag)

    #@classmethod
    #def is_gpu_dynamics_enabled(cls, physics_scene: str = None) -> bool:
    #    """Checks if Gpu Dynamics is enabled.

    #    Args:
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

    #    Returns:
    #        bool: True if Gpu Dynamics is enabled, otherwise False.
    #    """
    #    physx_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #    return physx_scene_api.GetEnableGPUDynamicsAttr().Get()

    # ------------------------------------------------------------------------------------------------------------------    

    @classmethod
    def enable_fabric(cls, enable):
        """Enables or disables physics fabric integration and associated settings.

        Args:
            enable: Whether to enable or disable fabric.
        """
        manager = omni.kit.app.get_app().get_extension_manager()
        fabric_was_enabled = manager.is_extension_enabled("omni.physx.fabric")
        if not fabric_was_enabled and enable:
            manager.set_extension_enabled_immediate("omni.physx.fabric", True)
        elif fabric_was_enabled and not enable:
            manager.set_extension_enabled_immediate("omni.physx.fabric", False)
        SimulationManager._carb_settings.set_bool("/physics/updateToUsd", not enable)
        SimulationManager._carb_settings.set_bool("/physics/updateParticlesToUsd", not enable)
        SimulationManager._carb_settings.set_bool("/physics/updateVelocitiesToUsd", not enable)
        SimulationManager._carb_settings.set_bool("/physics/updateForceSensorsToUsd", not enable)

    @classmethod
    def is_fabric_enabled(cls, enable):
        """Checks if fabric is enabled.

        Args:
            enable: Whether to check if fabric is enabled.

        Returns:
            bool: True if fabric is enabled, otherwise False.
        """
        return omni.kit.app.get_app().get_extension_manager().is_extension_enabled("omni.physx.fabric")

    # ------------------------------------------------------------------------------------------------------------------    

    # TODO: Removing this as we will only set the settings on initialization.
    #@classmethod
    #def set_solver_type(cls, solver_type: str, physics_scene: str = None) -> None:
    #    """solver used for simulation.

    #    Args:
    #        solver_type (str): can be "TGS" or "PGS".
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
    #    """
    #    if solver_type not in ["TGS", "PGS"]:
    #        raise Exception("Solver type {} is not supported".format(solver_type))
    #    if physics_scene is None:
    #        for path, physx_scene_api in SimulationManager._physics_scene_apis.items():
    #            if not physx_scene_api.GetPrim().IsValid():
    #                continue
    #            if physx_scene_api.GetSolverTypeAttr().Get() is None:
    #                physx_scene_api.CreateSolverTypeAttr(solver_type)
    #            else:
    #                physx_scene_api.GetSolverTypeAttr().Set(solver_type)
    #    else:
    #        physx_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #        if physx_scene_api.GetSolverTypeAttr().Get() is None:
    #            physx_scene_api.CreateSolverTypeAttr(solver_type)
    #        else:
    #            physx_scene_api.GetSolverTypeAttr().Set(solver_type)

    #@classmethod
    #def get_solver_type(cls, physics_scene: str = None) -> str:
    #    """Gets current solver type.

    #    Args:
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

    #    Returns:
    #        str: solver used for simulation.
    #    """
    #    physx_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #    return physx_scene_api.GetSolverTypeAttr().Get()

    # ------------------------------------------------------------------------------------------------------------------

    # TODO: Removing this, this is oddly specific, we will use string sets on initialization.
    #@classmethod
    #def enable_stablization(cls, flag: bool, physics_scene: str = None) -> None:
    #    """Enables additional stabilization pass in the solver.

    #    Args:
    #        flag (bool): enables or disables stabilization on the PhysicsScene
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.
    #    """
    #    if physics_scene is None:
    #        for path, physx_scene_api in SimulationManager._physics_scene_apis.items():
    #            if not physx_scene_api.GetPrim().IsValid():
    #                continue
    #            if physx_scene_api.GetEnableStabilizationAttr().Get() is None:
    #                physx_scene_api.CreateEnableStabilizationAttr(flag)
    #            else:
    #                physx_scene_api.GetEnableStabilizationAttr().Set(flag)
    #    else:
    #        physx_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #        if physx_scene_api.GetEnableStabilizationAttr().Get() is None:
    #            physx_scene_api.CreateEnableStabilizationAttr(flag)
    #        else:
    #            physx_scene_api.GetEnableStabilizationAttr().Set(flag)

    #@classmethod
    #def is_stablization_enabled(cls, physics_scene: str = None) -> bool:
    #    """Checks if stabilization is enabled.

    #    Args:
    #        physics_scene (str, optional): physics scene prim path.

    #    Raises:
    #        Exception: If the prim path registered in context doesn't correspond to a valid prim path currently.

    #    Returns:
    #        bool: True if stabilization is enabled, otherwise False.
    #    """
    #    physx_scene_api = SimulationManager._get_physics_scene_api(physics_scene=physics_scene)
    #    return physx_scene_api.GetEnableStabilizationAttr().Get()

    # ------------------------------------------------------------------------------------------------------------------

    # TODO: Removing this as the callbacks handling are moved to the SimulationContext class.
    #@classmethod
    #def register_callback(cls, callback: callable, event, order: int = 0, name: str = None):
    #    """Registers a callback to be triggered when a specific event occurs.

    #    Args:
    #        callback: The callback function to register.
    #        event: The event to trigger the callback.
    #        order: The order in which the callback should be triggered.
    #        name: The name of the callback.

    #    Returns:
    #        int: The ID of the callback.
    #    """
    #    proxy_needed = False
    #    if hasattr(callback, "__self__"):
    #        proxy_needed = True
    #        callback_name = callback.__name__
    #    callback_id = SimulationManager._simulation_manager_interface.get_callback_iter()
    #    if event in [
    #        IsaacEvents.PHYSICS_WARMUP,
    #        IsaacEvents.PHYSICS_READY,
    #        IsaacEvents.POST_RESET,
    #        IsaacEvents.SIMULATION_VIEW_CREATED,
    #    ]:
    #        if proxy_needed:
    #            SimulationManager._callbacks[callback_id] = SimulationManager._message_bus.observe_event(
    #                event_name=event.value,
    #                order=order,
    #                on_event=lambda event, obj=weakref.proxy(callback.__self__): getattr(obj, callback_name)(event),
    #                observer_name=f"SimulationManager._callbacks.{event.value}",
    #            )
    #        else:
    #            SimulationManager._callbacks[callback_id] = SimulationManager._message_bus.observe_event(
    #                event_name=event.value,
    #                order=order,
    #                on_event=callback,
    #                observer_name=f"SimulationManager._callbacks.{event.value}",
    #            )
    #    elif event == IsaacEvents.PRIM_DELETION:
    #        if proxy_needed:
    #            SimulationManager._simulation_manager_interface.register_deletion_callback(
    #                lambda event, obj=weakref.proxy(callback.__self__): getattr(obj, callback_name)(event)
    #            )
    #        else:
    #            SimulationManager._simulation_manager_interface.register_deletion_callback(callback)
    #    elif event == IsaacEvents.POST_PHYSICS_STEP:
    #        if proxy_needed:
    #            SimulationManager._callbacks[callback_id] = (
    #                SimulationManager._physx_interface.subscribe_physics_on_step_events(
    #                    lambda step_dt, obj=weakref.proxy(callback.__self__): (
    #                        getattr(obj, callback_name)(step_dt) if SimulationManager._simulation_view_created else None
    #                    ),
    #                    pre_step=False,
    #                    order=order,
    #                )
    #            )
    #        else:
    #            SimulationManager._callbacks[callback_id] = (
    #                SimulationManager._physx_interface.subscribe_physics_on_step_events(
    #                    lambda step_dt: callback(step_dt) if SimulationManager._simulation_view_created else None,
    #                    pre_step=False,
    #                    order=order,
    #                )
    #            )
    #    elif event == IsaacEvents.PRE_PHYSICS_STEP:
    #        if proxy_needed:
    #            SimulationManager._callbacks[callback_id] = (
    #                SimulationManager._physx_interface.subscribe_physics_on_step_events(
    #                    lambda step_dt, obj=weakref.proxy(callback.__self__): (
    #                        getattr(obj, callback_name)(step_dt) if SimulationManager._simulation_view_created else None
    #                    ),
    #                    pre_step=True,
    #                    order=order,
    #                )
    #            )
    #        else:
    #            SimulationManager._callbacks[callback_id] = (
    #                SimulationManager._physx_interface.subscribe_physics_on_step_events(
    #                    lambda step_dt: callback(step_dt) if SimulationManager._simulation_view_created else None,
    #                    pre_step=True,
    #                    order=order,
    #                )
    #            )
    #    elif event == IsaacEvents.TIMELINE_STOP:
    #        if proxy_needed:
    #            SimulationManager._callbacks[
    #                callback_id
    #            ] = SimulationManager._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
    #                int(omni.timeline.TimelineEventType.STOP),
    #                lambda event, obj=weakref.proxy(callback.__self__): getattr(obj, callback_name)(event),
    #                order=order,
    #                name=name,
    #            )
    #        else:
    #            SimulationManager._callbacks[
    #                callback_id
    #            ] = SimulationManager._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
    #                int(omni.timeline.TimelineEventType.STOP), callback, order=order, name=name
    #            )
    #    else:
    #        raise Exception("{} event doesn't exist for callback registering".format(event))
    #    SimulationManager._simulation_manager_interface.set_callback_iter(callback_id + 1)
    #    return callback_id

    # ------------------------------------------------------------------------------------------------------------------

    # TODO: Removing this as the callbacks handling are moved to the SimulationContext class.
    #@classmethod
    #def deregister_callback(cls, callback_id):
    #    """Deregisters a callback which was previously registered using register_callback.

    #    Args:
    #        callback_id: The ID of the callback returned by register_callback to deregister.
    #    """
    #    if callback_id in SimulationManager._callbacks:
    #        del SimulationManager._callbacks[callback_id]
    #    elif SimulationManager._simulation_manager_interface.deregister_callback(callback_id):
    #        return
    #    else:
    #        raise Exception("callback with id {} doesn't exist to be deregistered".format(callback_id))

    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def enable_usd_notice_handler(cls, flag):
        """Enables or disables the usd notice handler.

        Args:
            flag: Whether to enable or disable the handler.
        """
        SimulationManager._simulation_manager_interface.enable_usd_notice_handler(flag)
        return

    @classmethod
    def enable_fabric_usd_notice_handler(cls, stage_id, flag):
        """Enables or disables the fabric usd notice handler.

        Args:
            stage_id: The stage ID to enable or disable the handler for.
            flag: Whether to enable or disable the handler.
        """
        SimulationManager._simulation_manager_interface.enable_fabric_usd_notice_handler(stage_id, flag)
        return

    @classmethod
    def is_fabric_usd_notice_handler_enabled(cls, stage_id):
        """Checks if fabric usd notice handler is enabled.

        Args:
            stage_id: The stage ID to check.

        Returns:
            bool: True if fabric usd notice handler is enabled, otherwise False.
        """
        return SimulationManager._simulation_manager_interface.is_fabric_usd_notice_handler_enabled(stage_id)

    # ------------------------------------------------------------------------------------------------------------------

    # TODO: Removing this as we will not use asynchronous stuff.
    #@classmethod
    #def assets_loading(cls) -> bool:
    #    """Checks if textures are loaded.

    #    Returns:
    #        bool: True if textures are loading and not done yet, otherwise False.
    #    """
    #    return not SimulationManager._assets_loaded

    # ------------------------------------------------------------------------------------------------------------------

    # TODO: Removing these as the callbacks handling are moved to the SimulationContext class.
    # Public API methods for enabling/disabling callbacks
    #@classmethod
    #def enable_warm_start_callback(cls, enable: bool = True) -> None:
    #    """Enable or disable the warm start callback.

    #    Args:
    #        enable: Whether to enable the callback.
    #    """
    #    cls._callbacks_enabled["warm_start"] = enable
    #    if enable:
    #        cls._setup_warm_start_callback()
    #    else:
    #        if cls._warm_start_callback is not None:
    #            cls._warm_start_callback = None

    #@classmethod
    #def enable_on_stop_callback(cls, enable: bool = True) -> None:
    #    """Enable or disable the on stop callback.

    #    Args:
    #        enable: Whether to enable the callback.
    #    """
    #    cls._callbacks_enabled["on_stop"] = enable
    #    if enable:
    #        cls._setup_on_stop_callback()
    #    else:
    #        if cls._on_stop_callback is not None:
    #            cls._on_stop_callback = None

    #@classmethod
    #def enable_post_warm_start_callback(cls, enable: bool = True) -> None:
    #    """Enable or disable the post warm start callback.

    #    Args:
    #        enable: Whether to enable the callback.
    #    """
    #    cls._callbacks_enabled["post_warm_start"] = enable
    #    if enable:
    #        cls._setup_post_warm_start_callback()
    #    else:
    #        if cls._post_warm_start_callback is not None:
    #            cls._post_warm_start_callback = None

    #@classmethod
    #def enable_stage_open_callback(cls, enable: bool = True) -> None:
    #    """Enable or disable the stage open callback.
    #    Note: This also enables/disables the assets loading and loaded callbacks. If disabled, assets_loading() will always return True.

    #    Args:
    #        enable: Whether to enable the callback.
    #    """
    #    cls._callbacks_enabled["stage_open"] = enable
    #    if enable:
    #        cls._setup_stage_open_callback()
    #    else:
    #        if cls._stage_open_callback is not None:
    #            cls._stage_open_callback = None
    #            # Reset assets loading and loaded callbacks
    #            cls._assets_loaded = True
    #            cls._assets_loading_callback = None
    #            cls._assets_loaded_callback = None

    ## Convenience methods for bulk operations
    #@classmethod
    #def enable_all_default_callbacks(cls, enable: bool = True) -> None:
    #    """Enable or disable all default callbacks.
    #    Default callbacks are: warm_start, on_stop, post_warm_start, stage_open.

    #    Args:
    #        enable: Whether to enable all callbacks.
    #    """
    #    cls.enable_warm_start_callback(enable)
    #    cls.enable_on_stop_callback(enable)
    #    cls.enable_post_warm_start_callback(enable)
    #    cls.enable_stage_open_callback(enable)

    #@classmethod
    #def is_default_callback_enabled(cls, callback_name: str) -> bool:
    #    """Check if a specific default callback is enabled.
    #    Default callbacks are: warm_start, on_stop, post_warm_start, stage_open.

    #    Args:
    #        callback_name: Name of the callback to check.

    #    Returns:
    #        Whether the callback is enabled.
    #    """
    #    return cls._callbacks_enabled.get(callback_name, False)

    #@classmethod
    #def get_default_callback_status(cls) -> dict:
    #    """Get the status of all default callbacks.
    #    Default callbacks are: warm_start, on_stop, post_warm_start, stage_open.

    #    Returns:
    #        Dictionary with callback names and their enabled status.
    #    """
    #    return cls._callbacks_enabled.copy()
