#!/usr/bin/env python3

import argparse
import copy
import numpy as np
import os
from pathlib import Path
import pdb

from tqdm import tqdm
import uuid

from ab3dmot import AB3DMOT

import argoverse
from argoverse.data_loading.object_label_record import json_label_dict_to_obj_record
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.utils.se2 import SE2
from transform_utils import (
    yaw_to_quaternion3d, 
    se2_to_yaw, 
    get_B_SE2_A,
    rotmat2d
)
from json_utils import read_json_file, save_json_dict


def check_mkdir(dirpath):
    """ """
    if not Path(dirpath).exists():
        os.makedirs(dirpath, exist_ok=True)



class UUIDGeneration():
    def __init__(self):
        self.mapping = {}
    def get_uuid(self,seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex 
        return self.mapping[seed]
uuid_gen = UUIDGeneration()


def yaw_from_bbox_corners(det_corners: np.ndarray) -> float:
    """
    Use basic trigonometry on cuboid to get orientation angle.

        Args:
        -   det_corners: corners of bounding box

        Returns:
        -   yaw
    """
    p1 = det_corners[1]
    p5 = det_corners[5]
    dy = p1[1] - p5[1]
    dx = p1[0] - p5[0]
    # the orientation angle of the car
    yaw = np.arctan2(dy, dx)
    return yaw


def run_ab3dmot(
    classname: str,
    pose_dir: str,
    dets_dump_dir: str,
    tracks_dump_dir: str,
    max_age: int = 3,
    min_hits: int = 1,
    min_conf: float = 0.3
    ) -> None:
    """
    #path to argoverse tracking dataset test set, we will add our predicted labels into per_sweep_annotations_amodal/ 
    #inside this folder

    Filtering occurs in the city frame, not the egovehicle frame.

        Args:
        -   classname: string, either 'VEHICLE' or 'PEDESTRIAN'
        -   pose_dir: string
        -   dets_dump_dir: string
        -   tracks_dump_dir: string
        -   max_age: integer
        -   min_hits: integer

        Returns:
        -   None
    """
    dl = SimpleArgoverseTrackingDataLoader(data_dir=pose_dir, labels_dir=dets_dump_dir)

    for log_id in tqdm(dl.sdb.get_valid_logs()):
        print(log_id)
        labels_folder = dets_dump_dir + "/" + log_id + "/per_sweep_annotations_amodal/"
        lis = os.listdir(labels_folder)
        lidar_timestamps = [ int(file.split(".")[0].split("_")[-1]) for file in lis]
        lidar_timestamps.sort()
        previous_frame_bbox = []
        ab3dmot = AB3DMOT(max_age=max_age,min_hits=min_hits)
        print(labels_folder)
        tracked_labels_copy = []
        for j, current_lidar_timestamp in enumerate(lidar_timestamps):
            #print(current_lidar_timestamp)
            dets = dl.get_labels_at_lidar_timestamp(log_id, current_lidar_timestamp)
            #print(f'There are {len(dets)} detections!')
            
            dets_copy = dets
            transforms = []
            
            city_SE3_egovehicle = dl.get_city_to_egovehicle_se3(log_id, current_lidar_timestamp)
            egovehicle_SE3_city = city_SE3_egovehicle.inverse()
            transformed_labels = []
            for l_idx, l in enumerate(dets):

                if l['label_class'] != classname:
                    # will revisit in other tracking pass
                    continue
                if l["score"] < min_conf:
                    # print('Skipping det with confidence ', l["score"])
                    continue
                det_obj = json_label_dict_to_obj_record(l)
                det_corners_egovehicle_fr = det_obj.as_3d_bbox()
                
                transforms += [city_SE3_egovehicle]
                if city_SE3_egovehicle is None:
                    print('Was None')

                # convert detection from egovehicle frame to city frame
                det_corners_city_fr = city_SE3_egovehicle.transform_point_cloud(det_corners_egovehicle_fr)
                ego_xyz = np.mean(det_corners_city_fr, axis=0)

                yaw = yaw_from_bbox_corners(det_corners_city_fr)
                transformed_labels += [ [ego_xyz[0], ego_xyz[1], ego_xyz[2], yaw, l["length"],l["width"],l["height"]] ]

            if len(transformed_labels) > 0:
                transformed_labels = np.array(transformed_labels)
            else:
                transformed_labels = np.empty((0,7))
            
            dets_all = {
                "dets":transformed_labels,
                "info": np.zeros(transformed_labels.shape)
            }

            # perform measurement update in the city frame.
            dets_with_object_id = ab3dmot.update(dets_all)

            tracked_labels = []
            for det in dets_with_object_id:
                # move city frame tracks back to ego-vehicle frame
                xyz_city = np.array([det[0].item(), det[1].item(), det[2].item()]).reshape(1,3)
                city_yaw_object = det[3]
                city_se2_object = SE2(rotation=rotmat2d(city_yaw_object), translation=xyz_city.squeeze()[:2])
                city_se2_egovehicle, city_yaw_ego = get_B_SE2_A(city_SE3_egovehicle)
                ego_se2_city = city_se2_egovehicle.inverse()
                egovehicle_se2_object = ego_se2_city.right_multiply_with_se2(city_se2_object)

                # recreate all 8 points
                # transform them
                # compute yaw from 8 points once more
                egovehicle_SE3_city = city_SE3_egovehicle.inverse()
                xyz_ego = egovehicle_SE3_city.transform_point_cloud(xyz_city).squeeze()
                # update for new yaw
                # transform all 8 points at once, then compute yaw on the fly
       
                ego_yaw_obj = se2_to_yaw(egovehicle_se2_object)
                qx,qy,qz,qw = yaw_to_quaternion3d(ego_yaw_obj)
                tracked_labels.append({
                "center": {"x": xyz_ego[0], "y": xyz_ego[1], "z": xyz_ego[2]},
                "rotation": {"x": qx , "y": qy, "z": qz , "w": qw},
                "length": det[4],
                "width": det[5],
                "height": det[6],
                "track_label_uuid": uuid_gen.get_uuid(det[7]),
                 "timestamp": current_lidar_timestamp ,
                    "label_class": classname
                })

            tracked_labels_copy = copy.deepcopy(tracked_labels)

            label_dir = os.path.join(tracks_dump_dir, log_id, "per_sweep_annotations_amodal")    
            check_mkdir(label_dir)
            json_fname = f"tracked_object_labels_{current_lidar_timestamp}.json"
            json_fpath = os.path.join(label_dir, json_fname) 
            if Path(json_fpath).exists():
                # accumulate tracks of another class together
                prev_tracked_labels = read_json_file(json_fpath)
                tracked_labels.extend(prev_tracked_labels)
            
            save_json_dict(json_fpath, tracked_labels)


if __name__ == '__main__':
    """
    Run the tracker. The tracking is performed in the city frame, but the
    tracks will be dumped into the egovehicle frame for evaluation.
    2d IoU only is used for data association.
    
    Note:
        "max_age" denotes maximum allowed lifespan of a track (in timesteps of 100 ms) 
        since it was last updated with an associated measurement.

    Argparse args:
    -   split: dataset split
    -   max_age: max allowed track age since last measurement update
    -   min_hits: minimum number of required hits for track birth
    -   pose_dir: should be path to raw log files e.g.
            '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/test' or
            '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val/argoverse-tracking/val'
    -   dets_dataroot: should be path to 3d detections e.g.
            '/Users/johnlamb/Downloads/argoverse_detections_2020'
    -   tracks_dump_dir: where to dump the generated tracks
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, help="val or test")
    parser.add_argument("--max_age", type=int, default=15, 
            help="max allowed track age since last measurement update")
    parser.add_argument("--min_hits", type=int, default=5, 
        help="minimum number of required hits for track birth")

    parser.add_argument("--dets_dataroot", type=str, 
        required=True, help="path to 3d detections")

    parser.add_argument("--pose_dir", type=str, 
        required=True, help="path to raw log data (including pose data) for validation or test set")

    parser.add_argument("--tracks_dump_dir", type=str,
        default='temp_files',
        help="path to dump generated tracks (as .json files)")
    parser.add_argument("--min_conf", type=float,
        default=0.3,
        help="minimum allowed confidence for 3d detections to be considered valid")

    args = parser.parse_args()
    # tracks will be dumped into a subfolder of this name
    save_dirname = f'{args.split}-split-track-preds'
    save_dirname += f'-maxage{args.max_age}-minhits{args.min_hits}-conf{args.min_conf}'

    if args.split == 'train':
        args.dets_dataroot += '/training'
    elif args.split == 'val':
        args.dets_dataroot += '/validation'
    elif args.split == 'test':
        args.dets_dataroot += '/testing'

    args.tracks_dump_dir = f'{args.tracks_dump_dir}/{save_dirname}'

    # Run tracker over vehicle detections separately from ped. detections
    for classname in ['VEHICLE', 'PEDESTRIAN']:
        run_ab3dmot(
            classname,
            args.pose_dir,
            args.dets_dataroot,
            args.tracks_dump_dir,
            max_age=args.max_age,
            min_hits=args.min_hits,
            min_conf=args.min_conf
        )

