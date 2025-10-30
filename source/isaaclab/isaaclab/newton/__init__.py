from enum import IntEnum

class Frontend(IntEnum):
    WARP = 0
    TORCH = 1

class QuaternionFormat(IntEnum):
    XYZW = 0
    WXYZ = 1