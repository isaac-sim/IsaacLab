import math
from typing import Dict, Tuple


def swerve_isosceles_ik(
    vx: float,
    vy: float,
    wz: float,
    L1: float,
    d: float,
    w: float,
    R: float,
) -> Dict[str, Dict[str, float]]:
    """
    Inverse kinematics for a 3-module swerve (independent steering + drive)
    in an isosceles layout with +x pointing to Wheel 1.

    Geometry (meters):
        r1 = [ L1,  0 ]
        r2 = [ -d,  w ]
        r3 = [ -d, -w ]

    Inputs:
        vx    : chassis linear velocity in x (robot frame) [m/s]
        vy    : chassis linear velocity in y (robot frame) [m/s]
        wz    : chassis yaw rate about +z [rad/s] (CCW +)
        L1    : distance from center to Wheel 1 along +x [m]
        d     : x offset magnitude of base wheels (W2/W3) at x = -d [m]
        w     : half spacing of base wheels (W2/W3) at y = ±w [m]
        R     : wheel radius [m] (for converting linear speed to wheel angular speed)

    Outputs (per wheel i in {1,2,3}):
        angle_rad : steering angle α_i = atan2(v_i_y, v_i_x) [rad] (−π, π]
        angle_deg : same, degrees
        speed     : wheel linear speed s_i = ||v_i|| [m/s]
        omega     : wheel angular speed φ̇_i = s_i / R [rad/s]

    Equations (standard rigid-body velocity addition):
        v_i = [vx, vy]^T + wz * [-y_i, x_i]^T
        α_i = atan2(v_i_y, v_i_x)
        s_i = sqrt(v_i_x^2 + v_i_y^2)
        φ̇_i = s_i / R

    References:
      - WPILib swerve kinematics (maps chassis speeds → module angles & speeds). :contentReference[oaicite:0]{index=0}
      - Rigid-body relation v_P = v_O + ω × r (planar form used above). :contentReference[oaicite:1]{index=1}
      - Community derivations / implementation notes (angle optimization, etc.). :contentReference[oaicite:2]{index=2}
    """

    def module_state(xi: float, yi: float) -> Tuple[float, float, float, float]:
        vix = vx + wz * (-yi)
        viy = vy + wz * ( xi)
        angle = math.atan2(viy, vix)               # [-pi, pi]
        speed = math.hypot(vix, viy)
        omega = speed / R if R > 0 else float("inf")
        return angle, math.degrees(angle), speed, omega

    # Module positions (isosceles)
    r = {
        "wheel1": ( L1,  0.0),
        "wheel2": (-d ,  w ),
        "wheel3": (-d , -w ),
    }

    out = {}
    for name, (xi, yi) in r.items():
        angle_rad, angle_deg, speed, omega = module_state(xi, yi)
        out[name] = {
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
            "speed": speed,
            "omega": omega,
        }
    return out

