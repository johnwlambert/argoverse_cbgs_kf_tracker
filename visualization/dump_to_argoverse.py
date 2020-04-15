#!/usr/bin/env python3

import copy
import glob
import imageio
import json
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pdb
import pickle
from scipy.spatial.transform import Rotation
import sys
from tqdm import tqdm
from typing import Any, Iterable, List, Mapping, Sequence, Tuple, Union

from argoverse.data_loading.object_label_record import json_label_dict_to_obj_record
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.calibration import CameraConfig
from argoverse.utils.city_visibility_utils import clip_point_cloud_to_visible_region


from transform_utils import (
    get_B_SE2_A, 
    rotMatZ_3D,
    yaw_to_quaternion3d
)

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

#: Any numeric type
Number = Union[int, float]

# jigger lane pixel values by [-10,10] range
LANE_COLOR_NOISE = 20


def get_lane_yaw(avm, obj_center, city_SE3_egovehicle, city_name):
    """
    """
    # should be a query at the object's location, not at the egovehicles location

    query_xy_city_coords = city_SE3_egovehicle.transform_point_cloud(obj_center.reshape(1,3))

    query_xy_city_coords = query_xy_city_coords.reshape(3,)[:2]
    #query_xy_city_coords = city_SE3_egovehicle.translation[:2]

    vec, conf = avm.get_lane_direction(query_xy_city_coords, city_name)
    city_yaw_lane = np.arctan2(vec[1],vec[0])

    _, city_yaw_egovehicle = get_B_SE2_A(city_SE3_egovehicle)

    yaw = - city_yaw_egovehicle + city_yaw_lane
    return yaw


def read_pkl_file(fpath):
    """ """
    with open(fpath, "rb") as f:
        return pickle.load(f)


def dump_to_argoverse(
    use_map_lane_tangent, df, path, files, data_dir, pred_dump_dir, conf_thresh = 0.5
) -> List[str]:
    """
    We bring the 3D points into each camera coordinate system, and do the clipping there in 3D.

    Args:
        log_ids: A list of log IDs
        max_num_images_to_render: maximum numbers of images to render.
        data_dir: path to dataset with the latest data
        
        : Output directory
        motion_compensate: Whether to motion compensate when projecting

    Returns:
        saved_img_fpaths
    """
    avm = ArgoverseMap()
    dl = SimpleArgoverseTrackingDataLoader(data_dir=data_dir, labels_dir=data_dir)

    image_ids = [int(Path(file).stem) for file in files]
    logs_ids = []
    for image_id in image_ids:
        logs_ids.append(df[df["index"] == image_id].iloc[0].log_id)
    
    for image_id,file,log_id in tqdm(zip(image_ids,files,logs_ids)):
        print(log_id)

        row = df[df["index"] == image_id].iloc[0]
        lidar_timestamp = row["timestamp"]
        lidar_timestamp = int(lidar_timestamp)
        labels_pickle = read_pkl_file(f'{path}/{Path(file).name}')

        city_name = dl.get_city_name(log_id)
        city_SE3_egovehicle = dl.get_city_to_egovehicle_se3(log_id, lidar_timestamp)

        selected = labels_pickle["score"] >= conf_thresh
        centers = labels_pickle["location_lidar"][selected]
        dimensions = labels_pickle["dimensions_lidar"][selected]
        rotation_y = labels_pickle["rotation_y_lidar"][selected]
        labels = []
        number_of_instances = np.sum(selected)
        for i in range(number_of_instances):
            yaw_est = rotation_y[i]-(np.pi/2)
            if use_map_lane_tangent:
                try: # try to use the map
                    yaw_map = get_lane_yaw(avm, centers[i], city_SE3_egovehicle, city_name)
                    
                    if np.absolute(yaw_map - yaw_est) < (np.pi/4): # if huge disagreement, sketchy
                        yaw = yaw_map
                    else:
                        yaw = yaw_est
                except: # map failed, revert back
                    yaw = yaw_est
            else: # not allowed to use the map
                yaw = yaw_est
            
            qx,qy,qz,qw = yaw_to_quaternion3d(yaw)

            #qx,qy,qz,qw = Rotation.from_euler('z', rotation_y[i]-np.pi/2).as_quat() 
            labels.append({
            "center": {"x": centers[i,0], "y": centers[i,1], "z":centers[i,2] },
            "rotation": {"x": qx , "y": qy, "z": qz , "w": qw},
            "length" :dimensions[i,1],
            "width": dimensions[i,0],
            "height": dimensions[i,2],
            "track_label_uuid": None,
             "timestamp": lidar_timestamp ,
                "label_class": "VEHICLE"
            })

        label_objects = [json_label_dict_to_obj_record(label) for label_idx, label in enumerate(labels)]
        data_dir = data_dir

        log = log_id

        label_dir = os.path.join(pred_dump_dir, log, "per_sweep_annotations_amodal")

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        labels_json_data = []

        timestamp = lidar_timestamp

        for label in label_objects:
            json_data = {
                "center": {"x": label.translation[0].item(), "y": label.translation[1].item(), "z": label.translation[2].item()},
                "rotation": {
                    "w": label.quaternion[0].item(),
                    "x": label.quaternion[1].item(),
                    "y": label.quaternion[2].item(),
                    "z": label.quaternion[3].item(),
                },
                "length": label.length.item(),
                "width": label.width.item(),
                "height": label.height.item(),
                "occlusion": 0,
                "tracked": True,
                "timestamp": timestamp,
                "label_class": label.label_class,
                "track_label_uuid": label.track_id,
            }
            labels_json_data.append(json_data)
        fn = f"tracked_object_labels_{timestamp}.json"
        with open(os.path.join(label_dir, fn), "w") as json_file:
            json.dump(labels_json_data, json_file)




if __name__ == "__main__":
    # Parse command line arguments
    # pkl_path ="/srv/scratch/svanga3/pred_dump_for_plots2019-11-26_20:58:56/" 
    # pkl_path = "pkl/"
    use_map_lane_tangent = True
    split = 'val'# 'test'# 
    if split == 'test':
        pkl_path = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/all_pkl_test_split'
        dataset_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/test'
        csv_fpath = "argo_to_kitti_map_test_submission.csv"
        #pred_dump_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/test-track-preds'
        #pred_dump_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/test-dets-preds-greedy-match-network-yaw'
        pred_dump_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/test-use-map-dets-2019-12-04'

    elif split == 'val':
        pkl_path = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/all_pkl_val'
        dataset_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val/argoverse-tracking/val'
        csv_fpath = 'argo_to_kitti_map_val.csv'
        #pred_dump_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val-track-preds'
        #pred_dump_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val-dets-preds-greedy-match-network-yaw'
        # pred_dump_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val-use-map-dets-2019-12-04'
        # pred_dump_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val-use-map-dets-2019-12-05-confpt4'
        pred_dump_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val-use-map-dets-2019-12-05-confpt30'
    else:
        print('Unknown split. Quitting...')
        quit()

    conf_thresh = 0.3
    df = pd.read_csv(csv_fpath)
    fpaths = glob.glob(f'{pkl_path}/*.pickle')
    fpaths = sorted(fpaths, key = lambda x: int(Path(x).stem) )  
    dump_to_argoverse(use_map_lane_tangent, df, pkl_path, fpaths, dataset_dir, pred_dump_dir, conf_thresh)


