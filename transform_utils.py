#!/usr/bin/env python3

import copy
from numba import jit
import numpy as np
import pdb
from scipy.spatial.transform import Rotation

from typing import Tuple

from argoverse.utils.se3 import SE3
from argoverse.utils.se2 import SE2


def rotmat2d(theta: float) -> np.ndarray:
    """
        Return rotation matrix corresponding to rotation theta.
        
        Args:
        -   theta: rotation amount in radians.
        
        Returns:
        -   R: 2 x 2 np.ndarray rotation matrix corresponding to rotation theta.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array(
        [
            [cos_theta, -sin_theta], 
            [sin_theta, cos_theta]
        ])
    return R


def get_B_SE2_A(B_SE3_A: SE3):
    """
        Can take city_SE3_egovehicle -> city_SE2_egovehicle
        Can take egovehicle_SE3_object -> egovehicle_SE2_object

        Doesn't matter if we stretch square by h,w,l since
        triangles will be similar regardless

        Args:
        -   B_SE3_A

        Returns:
        -   B_SE2_A
        -   B_yaw_A
    """
    x_corners = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners_A_frame = np.vstack((x_corners, y_corners, z_corners)).T

    corners_B_frame = B_SE3_A.transform_point_cloud(corners_A_frame)

    p1 = corners_B_frame[1]
    p5 = corners_B_frame[5]
    dy = p1[1] - p5[1]
    dx = p1[0] - p5[0]
    # the orientation angle of the car
    B_yaw_A = np.arctan2(dy, dx)

    t = B_SE3_A.transform_matrix[:2,3] # get x,y only
    B_SE2_A = SE2(
        rotation=rotmat2d(B_yaw_A),
        translation=t
    )
    return B_SE2_A, B_yaw_A



def se2_to_yaw(B_SE2_A):
    """
    Computes the pose vector v from a homogeneous transform A.
    Args:
    -   B_SE2_A
    Returns:
    -   v
    """
    R = B_SE2_A.rotation
    theta = np.arctan2(R[1,0], R[0,0])
    return theta


def yaw_to_quaternion3d(yaw: float) -> Tuple[float,float,float,float]:
    """
    Args:
    -   yaw: rotation about the z-axis

    Returns:
    -   qx,qy,qz,qw: quaternion coefficients
    """
    qx,qy,qz,qw = Rotation.from_euler('z', yaw).as_quat()
    return qx,qy,qz,qw



def test_yaw_to_quaternion3d():
    """
    """
    for i, yaw in enumerate(np.linspace(0, 3*np.pi, 50)):
        print(f'On iter {i}')
        dcm = rotMatZ_3D(yaw)
        qx,qy,qz,qw = Rotation.from_dcm(dcm).as_quat()

        qx_, qy_, qz_, qw_ = yaw_to_quaternion3d(yaw)
        print(qx_, qy_, qz_, qw_, ' vs ', qx,qy,qz,qw)
        assert np.allclose(qx, qx_, atol=1e-3)
        assert np.allclose(qy, qy_, atol=1e-3)
        assert np.allclose(qz, qz_, atol=1e-3)
        assert np.allclose(qw, qw_, atol=1e-3)


@jit       
def roty(t: float):
    """
    Compute rotation matrix about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    R = np.array(
        [
            [c,  0,  s],
            [0,  1,  0],
            [-s, 0,  c]
        ])
    return R


def rotMatZ_3D(yaw):
    """
        Args:
        -   tz

        Returns:
        -   rot_z
    """
    # c = np.cos(yaw)
    # s = np.sin(yaw)
    # rot_z = np.array(
    #     [
    #         [   c,-s, 0],
    #         [   s, c, 0],
    #         [   0, 0, 1 ]
    #     ])

    rot_z = Rotation.from_euler('z', yaw).as_dcm()
    return rot_z


def convert_3dbox_to_8corner(bbox3d_input: np.ndarray) -> np.ndarray:
    '''
        Args:
        -   bbox3d_input: Numpy array of shape (7,) representing
                tx,ty,tz,yaw,l,w,h. (tx,ty,tz,yaw) tell us how to
                transform points to get from the object frame to 
                the egovehicle frame.

        Returns:
        -   corners_3d: (8,3) array in egovehicle frame
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)
    yaw = bbox3d[3]
    t = bbox3d[:3]

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];

    # rotate and translate 3d bounding box
    corners_3d_obj_fr = np.vstack([x_corners,y_corners,z_corners]).T
    egovehicle_SE3_object = SE3(rotation=rotMatZ_3D(yaw), translation=t)
    corners_3d_ego_fr = egovehicle_SE3_object.transform_point_cloud(corners_3d_obj_fr)
    return corners_3d_ego_fr



if __name__ == '__main__':
    test_yaw_to_quaternion3d()




