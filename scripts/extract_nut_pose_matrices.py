#!/usr/bin/env python3
"""Reset the real robosuite nut_assembly env and print the actual GT pose matrices
(T_hole, T_grasp, T_peg, T_rel = inv(T_hole)@T_grasp, T_target = T_peg@T_rel) for a few
random episodes — the genuine numbers behind capx_nut_insertion_math.html.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=4, suppress=True)
from capx.envs.base import get_env


def to_T(p):
    pos, q_wxyz = np.asarray(p[:3], float), np.asarray(p[3:], float)
    T = np.eye(4)
    T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    T[:3, 3] = pos
    return T


def yaw(T):
    return round(float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))), 1)


def main(n=3):
    env = get_env("franka_robosuite_nut_assembly_low_level_visual", privileged=True, enable_render=False)
    for ep in range(n):
        env.reset()
        np_ = env.get_observation()["nut_poses"]
        T_hole = to_T(np_["square_nut"])          # 너트 중심(구멍)
        T_grasp = to_T(np_["square_nut_handle"])  # 손잡이 grasp
        T_peg = to_T(np_["square_peg"])           # peg
        T_rel = np.linalg.inv(T_hole) @ T_grasp
        T_target = T_peg @ T_rel
        print(f"\n================== EPISODE {ep} ==================")
        for name, T in [("T_hole(구멍)", T_hole), ("T_grasp(손잡이)", T_grasp),
                        ("T_peg", T_peg), ("T_rel=inv(T_hole)@T_grasp", T_rel),
                        ("T_target=T_peg@T_rel", T_target)]:
            print(f"--- {name}  (yaw={yaw(T)}°) ---")
            print(T)
        print(f"YAWS: hole={yaw(T_hole)}  grasp={yaw(T_grasp)}  peg={yaw(T_peg)}  "
              f"target={yaw(T_target)}  | 잡은채로(=grasp) vs 정렬(=target) 차이="
              f"{round(yaw(T_grasp)-yaw(T_target),1)}°")


if __name__ == "__main__":
    main()
