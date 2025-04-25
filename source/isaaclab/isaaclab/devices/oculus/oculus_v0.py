"Joystick controller using OculusReader input."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni
import time
import math
import ipdb
from ..device_base import DeviceBase

# from .oculus_reader import OculusReader
from .FPS_counter import FPSCounter
from .buttons_parser import parse_buttons
import threading
import os
from ppadb.client import Client as AdbClient
import sys

def eprint(*args, **kwargs):
    RED = "\033[1;31m"  
    sys.stderr.write(RED)
    print(*args, file=sys.stderr, **kwargs)
    RESET = "\033[0;0m"
    sys.stderr.write(RESET)

class OculusReader:
    def __init__(self,
            ip_address=None,
            port = 5555,
            APK_name='com.rail.oculus.teleop',
            print_FPS=False,
            run=True
        ):
        self.running = False
        self.last_transforms = {}
        self.last_buttons = {}
        self._lock = threading.Lock()
        self.tag = 'wE9ryARX'

        self.ip_address = ip_address
        self.port = port
        self.APK_name = APK_name
        self.print_FPS = print_FPS
        if self.print_FPS:
            self.fps_counter = FPSCounter()

        self.device = self.get_device()
        self.install(verbose=False)
        if run:
            self.run()

    def __del__(self):
        self.stop()

    def run(self):
        self.running = True
        self.device.shell('am start -n "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER')
        self.thread = threading.Thread(target=self.device.shell, args=("logcat -T 0", self.read_logcat_by_line))
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def get_network_device(self, client, retry=0):
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            os.system('adb devices')
            client.remote_connect(self.ip_address, self.port)
        device = client.device(self.ip_address + ':' + str(self.port))

        if device is None:
            if retry==1:
                os.system('adb tcpip ' + str(self.port))
            if retry==2:
                eprint('Make sure that device is running and is available at the IP address specified as the OculusReader argument `ip_address`.')
                eprint('Currently provided IP address:', self.ip_address)
                eprint('Run `adb shell ip route` to verify the IP address.')
                exit(1)
            else:
                self.get_device(client=client, retry=retry+1)
        return device

    def get_usb_device(self, client):
        try:
            devices = client.devices()
        except RuntimeError:
            os.system('adb devices')
            devices = client.devices()
        for device in devices:
            if device.serial.count('.') < 3:
                return device
        eprint('Device not found. Make sure that device is running and is connected over USB')
        eprint('Run `adb devices` to verify that the device is visible.')
        exit(1)

    def get_device(self):
        # Default is "127.0.0.1" and 5037
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.ip_address is not None:
            return self.get_network_device(client)
        else:
            return self.get_usb_device(client)

    def install(self, APK_path=None, verbose=True, reinstall=False):
        try:
            installed = self.device.is_installed(self.APK_name)
            if not installed or reinstall:
                if APK_path is None:
                    APK_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'APK', 'teleop-debug.apk')
                success = self.device.install(APK_path, test=True, reinstall=reinstall)
                installed = self.device.is_installed(self.APK_name)
                if installed and success:
                    print('APK installed successfully.')
                else:
                    eprint('APK install failed.')
            elif verbose:
                print('APK is already installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    def uninstall(self, verbose=True):
        try:
            installed = self.device.is_installed(self.APK_name)
            if installed:
                success = self.device.uninstall(self.APK_name)
                installed = self.device.is_installed(self.APK_name)
                if not installed and success:
                    print('APK uninstall finished.')
                    print('Please verify if the app disappeared from the list as described in "UNINSTALL.md".')
                    print('For the resolution of this issue, please follow https://github.com/Swind/pure-python-adb/issues/71.')
                else:
                    eprint('APK uninstall failed')
            elif verbose:
                print('APK is not installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    @staticmethod
    def process_data(string):
        try:
            transforms_string, buttons_string = string.split('&')
        except ValueError:
            return None, None
        split_transform_strings = transforms_string.split('|')
        transforms = {}
        for pair_string in split_transform_strings:
            transform = np.empty((4,4))
            pair = pair_string.split(':')
            if len(pair) != 2:
                continue
            left_right_char = pair[0] # is r or l
            transform_string = pair[1]
            values = transform_string.split(' ')
            c = 0
            r = 0
            count = 0
            for value in values:
                if not value:
                    continue
                transform[r][c] = float(value)
                c += 1
                if c >= 4:
                    c = 0
                    r += 1
                count += 1
            if count == 16:
                transforms[left_right_char] = transform
        buttons = parse_buttons(buttons_string)
        return transforms, buttons

    def extract_data(self, line):
        output = ''
        if self.tag in line:
            try:
                output += line.split(self.tag + ': ')[1]
            except ValueError:
                pass
        return output

    def get_transformations_and_buttons(self):
        with self._lock:
            return self.last_transforms, self.last_buttons

    def read_logcat_by_line(self, connection):
        file_obj = connection.socket.makefile()
        while self.running:
            try:
                line = file_obj.readline().strip()
                data = self.extract_data(line)
                if data:
                    transforms, buttons = OculusReader.process_data(data)
                    with self._lock:
                        self.last_transforms, self.last_buttons = transforms, buttons
                    if self.print_FPS:
                        self.fps_counter.getAndPrintFPS()
            except UnicodeDecodeError:
                pass
        file_obj.close()
        connection.close()





class OculusV0(DeviceBase):
    """A joystick controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a joystick controller for a bi-manual mobile manipulator.
    It uses the rail oculus reader to listen to joystick events and map them to robot's
    task-space commands.

    The command comprises of three parts:

    * delta control for mobile base: a 3D vector of (vx, vy, vz) in meters per second.
    * delta pose for each arms: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper for each arms: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= 
        Description                    Joystick event    
        ============================== ================= 
        Toggle gripperR (open/close)   RTr (index finger)
        Toggle gripperL (open/close)   LTr (index finger)
        Move along xy plane            leftJS xy position
        Rotate along yaw               RightJS x         
        Move right arm                 IK of right joystick SE(3) 4x4 matrix
        Move left arm                  IK of left joystick SE(3) 4x4 matrix
        ============================== ================= 
    """
    def __init__(self, pos_sensitivity: float = 1, rot_sensitivity: float = 1, base_sensitivity: float = 0.3):
        """
        Args:
            pos_sensitivity: Magnitude of input position command scaling for arms.
            rot_sensitivity: Magnitude of input rotation command scaling for arms.
            base_sensitivity: Magnitude of input command scaling for the mobile base.
        """
        # sensitivities
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.base_sensitivity = base_sensitivity

        # initialize OculusReader for joystick input
        self.oculus_reader = OculusReader()
        self.oculus_reader.initialize()  # start background threads if needed

        # set up yaw cumulative motion
        self._key_hold_start = {}  # Track when rightJS is moved
        self._base_z_accum = 0.0   # Accumulated vertical motion for base

        # command buffers
        self._close_gripper_left = False
        self._close_gripper_right = False
        self._delta_pos_left = np.zeros(3)  # (x, y, z) for left arm
        self._delta_rot_left = np.zeros(3)  # (roll, pitch, yaw) for left arm
        self._delta_pos_right = np.zeros(3)  # (x, y, z) for right arm
        self._delta_rot_right = np.zeros(3)  # (roll, pitch, yaw) for right arm
        self._delta_base = np.zeros(3)  # (x, y, yaw) for mobile base
        
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self.oculus_reader.stop()
    
    def reset(self):
        """Reset all commands."""
        self._close_gripper_left = False
        self._close_gripper_right = False
        self._delta_pos_left = np.zeros(3)
        self._delta_rot_left = np.zeros(3)
        self._delta_pos_right = np.zeros(3)
        self._delta_rot_right = np.zeros(3)
        self._delta_base = np.zeros(3)
        self._base_z_accum = 0.0
        self._key_hold_start = {}  # Reset key hold tracking

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Joystick Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"

    #[TODO: Reset event in add_callback]
    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func


    def advance(self) -> tuple[np.ndarray, bool, np.ndarray, bool, np.ndarray]:
        """
        Read joystick events and return command for BMM.

        Threshold:
            base xy: 0.1
            base z: 0.7

        Returns:
            A tuple containing the delta pose commands for left arm, right arm, and mobile base.
        """
        # fetch latest controller data
        transforms, buttons = self.oculus_reader.get_transformations_and_buttons()  # returns dict with 'leftJS', 'rightJS'
        
        # Delta pose for arms
        self._delta_pos_left += np.array([transforms["l"][0, 3], transforms["l"][1, 3], transforms["l"][2, 3]]) * self.pos_sensitivity
        self._delta_pos_right += np.array([transforms["r"][0, 3], transforms["l"][1, 3], transforms["l"][2, 3]]) * self.pos_sensitivity

        # Rotation for arms
        rot_vec_left = Rotation.from_matrix(transforms["l"][:3, :3]).as_rotvec()
        rot_vec_right = Rotation.from_matrix(transforms["r"][:3, :3]).as_rotvec()

        # Gripper
        if buttons['LTr']:
            self._close_gripper_left = not self._close_gripper_left
        if buttons['RTr']:
            self._close_gripper_right = not self._close_gripper_right

        # mobile base

        # yaw
        # check if the rightJS is moved to right
        if buttons['rightJS'][0] > 0.7  and ('counterclockwise' not in self._key_hold_start):
            self._delta_base[2] -= self.base_sensitivity
            self._key_hold_start['counterclockwise'] = time.time()

        # check if the rightJS is moved to left
        elif buttons['rightJS'][0] < -0.7 and ('clockwise' not in self._key_hold_start):
            self._delta_base[2] += self.base_sensitivity
            self._key_hold_start['clockwise'] = time.time()

        # check if the rightJS is returned to the center (similar to key realeased)
        elif buttons['rightJS'][0] == 0.0 & (('counterclockwise' in self._key_hold_start) or ()'clockwise' in self._key_hold_start)):
            # remove the key from the dictionary
            if 'counterclockwise' in self._key_hold_start:
                duration = time.time() - self._key_hold_start['counterclockwise']
                self._delta_base[2] += self.base_sensitivity * duration
                del self._key_hold_start['counterclockwise']
                
            if 'clockwise' in self._key_hold_start:
                duration = time.time() - self._key_hold_start['clockwise']
                self._delta_base[2] -= self.base_sensitivity * duration
                del self._key_hold_start['clockwise']
        # xy
        if buttons['leftJS'] != (0.0, 0.0):
            raw_x, raw_y = buttons['leftJS']  # x, y in [-1, 1]
            self._delta_base += raw_x * np.asarray([ math.cos(self._base_z_accum / 1.25),  math.sin(self._base_z_accum / 1.25), 0.0]) * self.base_sensitivity
            self._delta_base += raw_y * np.asarray([ math.sin(self._base_z_accum / 1.25),  -math.cos(self._base_z_accum / 1.25), 0.0]) * self.base_sensitivity

        # return the commands
        return (
            np.concatenate([self._delta_pos_left, rot_vec_left]),  # Left arm
            self._close_gripper_left,  # Left gripper
            np.concatenate([self._delta_pos_right, rot_vec_right]),  # Right arm
            self._close_gripper_right,  # Right gripper
            self._delta_base,  # Mobile base
        )
    #  ({'l': array([[-0.224735 ,  0.415998 , -0.881158 , -0.0416255],
    #    [ 0.937851 , -0.153066 , -0.311457 , -0.103081 ],
    #    [-0.264441 , -0.89639  , -0.355745 ,  0.0854378],
    #    [ 0.       ,  0.       ,  0.       ,  1.       ]]), 
       
    #    'r': array([[-0.753894 ,  0.601267 ,  0.264804 , -0.0686894],
    #    [-0.130283 ,  0.258232 , -0.957258 , -0.0819744],
    #    [-0.643948 , -0.756171 , -0.116345 ,  0.0297576],
    #    [ 0.       ,  0.       ,  0.       ,  1.       ]])}, {'A': False, 'B': False, 'RThU': True, 'RJ': False, 'RG': False, 'RTr': False, 'X': False, 'Y': False, 'LThU': True, 'LJ': False, 'LG': False, 'LTr': False, 'leftJS': (0.0, 0.0), 'leftTrig': (0.0,), 'leftGrip': (0.0,), 'rightJS': (0.0, 0.0), 'rightTrig': (0.0,), 'rightGrip': (0.0,)})

