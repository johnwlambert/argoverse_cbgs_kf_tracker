# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import argparse
import glob
import json
import logging
import os
import pathlib
from pathlib import Path
from typing import Any, Dict, List, TextIO, Tuple, Union

import motmetrics as mm
import numpy as np
from shapely.geometry.polygon import Polygon

from argoverse.evaluation.eval_utils import get_pc_inside_bbox, label_to_bbox, leave_only_roi_region
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

mh = mm.metrics.create()
logger = logging.getLogger(__name__)

_PathLike = Union[str, "os.PathLike[str]"]


def in_distance_range_pose(ego_center: np.ndarray, pose: np.ndarray, d_min: float, d_max: float) -> bool:
    """Determine if a pose is within distance range or not.

    Args:
        ego_center: ego center pose (zero if bbox is in ego frame).
        pose:  pose to test.
        d_min: minimum distance range
        d_max: maximum distance range

    Returns:
        A boolean saying if input pose is with specified distance range.
    """

    dist = float(np.linalg.norm(pose[0:3] - ego_center[0:3]))

    return dist > d_min and dist < d_max


def iou_polygon(poly1: Polygon, poly2: Polygon) -> float:
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return float(1 - inter / union)


def get_distance_iou_3d(x1: np.ndarray, x2: np.ndarray, name: str = "bbox") -> float:

    w1 = x1["width"]
    l1 = x1["length"]
    h1 = x1["height"]

    w2 = x2["width"]
    l2 = x2["length"]
    h2 = x2["height"]

    poly1 = Polygon([(-l1 / 2, -w1 / 2), (-l1 / 2, w1 / 2), (l1 / 2, w1 / 2), (l1 / 2, -w1 / 2)])
    poly2 = Polygon([(-l2 / 2, -w2 / 2), (-l2 / 2, w2 / 2), (l2 / 2, w1 / 2), (l2 / 2, -w2 / 2)])

    inter = poly1.intersection(poly2).area * min(h1, h2)
    union = w1 * l1 * h1 + w2 * l2 * h2 - inter
    score = 1 - inter / union

    return float(score)


def get_distance(x1: np.ndarray, x2: np.ndarray, name: str) -> float:
    """Get the distance between two poses, returns nan if distance is larger than detection threshold.

    Args:
        x1: first pose
        x2: second pose
        name: name of the field to test

    Returns:
        A distance value or NaN
    """
    if name == "centroid":
        dist = float(np.linalg.norm(x1[name][0:3] - x2[name][0:3]))
        return dist if dist < 2 else float(np.nan)
    elif name == "iou":
        return get_distance_iou_3d(x1, x2, name)
    elif name == "orientation":
        return float(
            min(np.abs(x1[name] - x2[name]), np.abs(np.pi + x1[name] - x2[name]), np.abs(-np.pi + x1[name] - x2[name]))
            * 180
            / np.pi
        )
    else:
        raise ValueError("Not implemented..")


def get_forth_vertex_rect(
    p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]
) -> Tuple[float, float]:
    x = p2[0] - p1[0] + p3[0]
    y = p3[1] - p1[1] + p2[1]
    return (x, y)


def eval_tracks(
    log_ids, 
    tracker_output_dir: _PathLike,
    dataset_dir: _PathLike,
    d_min: float,
    d_max: float,
    out_file: TextIO,
    centroid_method: str,
    category: str = "VEHICLE",
) -> None:
    """Evaluate tracking output.

    Args:
        path_tracker_output: list of path to tracker output, one for each log
        path_dataset: path to dataset
        d_min: minimum distance range
        d_max: maximum distance range
        out_file: output file object
        centroid_method: method for ground truth centroid estimation
    """
    acc_c = mm.MOTAccumulator(auto_id=True)
    acc_i = mm.MOTAccumulator(auto_id=True)
    acc_o = mm.MOTAccumulator(auto_id=True)

    ID_gt_all: List[str] = []

    count_all: int = 0

    for log_id in log_ids:
        path_tracker_output = f'{tracker_output_dir}/{log_id}'
        path_dataset = f'{dataset_dir}/{log_id}'

        path_track_data = sorted(glob.glob(path_tracker_output + '/per_sweep_annotations_amodal/*'))
        logger.info("log_id = %s", log_id)

        city_info_fpath = f"{path_dataset}/city_info.json"
        city_info = read_json_file(city_info_fpath)
        city_name = city_info["city_name"]
        logger.info("city name = %s", city_name)

        for ind_frame in range(len(path_track_data)):
            if ind_frame % 50 == 0:
                logger.info("%d/%d" % (ind_frame, len(path_track_data)))

            # import pdb; pdb.set_trace()
            timestamp_lidar = int(Path(path_track_data[ind_frame]).stem.split("_")[-1])
            path_gt = os.path.join(
                path_dataset, "per_sweep_annotations_amodal", f"tracked_object_labels_{timestamp_lidar}.json"
            )

            if not os.path.exists(path_gt):
                logger.warning("Missing ", path_gt)
                continue

            gt_data = read_json_file(path_gt)

            pose_data = read_json_file(f"{path_dataset}/poses/city_SE3_egovehicle_{timestamp_lidar}.json")
            rotation = np.array(pose_data["rotation"])
            translation = np.array(pose_data["translation"])
            ego_R = quat2rotmat(rotation)
            ego_t = translation
            egovehicle_to_city_se3 = SE3(rotation=ego_R, translation=ego_t)

            gt: Dict[str, Dict[str, Any]] = {}
            id_gts = []
            for i in range(len(gt_data)):

                if gt_data[i]["label_class"] != category:
                    continue

                bbox, orientation = label_to_bbox(gt_data[i])

                center = np.array([gt_data[i]["center"]["x"], gt_data[i]["center"]["y"], gt_data[i]["center"]["z"]])
                if bbox[3] > 0 and in_distance_range_pose(np.zeros(3), center, d_min, d_max):
                    track_label_uuid = gt_data[i]["track_label_uuid"]
                    gt[track_label_uuid] = {}
                    gt[track_label_uuid]["centroid"] = center

                    gt[track_label_uuid]["bbox"] = bbox
                    gt[track_label_uuid]["orientation"] = orientation
                    gt[track_label_uuid]["width"] = gt_data[i]["width"]
                    gt[track_label_uuid]["length"] = gt_data[i]["length"]
                    gt[track_label_uuid]["height"] = gt_data[i]["height"]

                    if track_label_uuid not in ID_gt_all:
                        ID_gt_all.append(track_label_uuid)

                    id_gts.append(track_label_uuid)

            tracks: Dict[str, Dict[str, Any]] = {}
            id_tracks: List[str] = []

            track_data = read_json_file(path_track_data[ind_frame])

            for track in track_data:
                key = track["track_label_uuid"]

                if track["label_class"] != category or track["height"] == 0:
                    continue

                center = np.array([track["center"]["x"], track["center"]["y"], track["center"]["z"]])
                bbox, orientation = label_to_bbox(track)
                if in_distance_range_pose(np.zeros(3), center, d_min, d_max):
                    tracks[key] = {}
                    tracks[key]["centroid"] = center
                    tracks[key]["bbox"] = bbox
                    tracks[key]["orientation"] = orientation
                    tracks[key]["width"] = track["width"]
                    tracks[key]["length"] = track["length"]
                    tracks[key]["height"] = track["height"]

                    id_tracks.append(key)

            dists_c: List[List[float]] = []
            dists_i: List[List[float]] = []
            dists_o: List[List[float]] = []
            for gt_key, gt_value in gt.items():
                gt_track_data_c: List[float] = []
                gt_track_data_i: List[float] = []
                gt_track_data_o: List[float] = []
                dists_c.append(gt_track_data_c)
                dists_i.append(gt_track_data_i)
                dists_o.append(gt_track_data_o)
                for track_key, track_value in tracks.items():
                    count_all += 1
                    gt_track_data_c.append(get_distance(gt_value, track_value, "centroid"))
                    gt_track_data_i.append(get_distance(gt_value, track_value, "iou"))
                    gt_track_data_o.append(get_distance(gt_value, track_value, "orientation"))

            acc_c.update(id_gts, id_tracks, dists_c)
            acc_i.update(id_gts, id_tracks, dists_i)
            acc_o.update(id_gts, id_tracks, dists_o)
    if count_all == 0:
        # fix for when all hypothesis is empty,
        # pymotmetric currently doesn't support this, see https://github.com/cheind/py-motmetrics/issues/49
        acc_c.update(id_gts, ["dummy_id"], np.ones(np.shape(id_gts)) * np.inf)
        acc_i.update(id_gts, ["dummy_id"], np.ones(np.shape(id_gts)) * np.inf)
        acc_o.update(id_gts, ["dummy_id"], np.ones(np.shape(id_gts)) * np.inf)

    summary = mh.compute(
        acc_c,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "mostly_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
        ],
        name="acc",
    )
    logger.info("summary = %s", summary)
    num_tracks = len(ID_gt_all)

    fn = os.path.basename(path_tracker_output)
    num_frames = summary["num_frames"][0]
    mota = summary["mota"][0] * 100
    motp_c = summary["motp"][0]
    idf1 = summary["idf1"][0]
    most_track = summary["mostly_tracked"][0] / num_tracks
    most_lost = summary["mostly_lost"][0] / num_tracks
    num_fp = summary["num_false_positives"][0]
    num_miss = summary["num_misses"][0]
    num_switch = summary["num_switches"][0]
    num_flag = summary["num_fragmentations"][0]

    acc_c.events.loc[acc_c.events.Type != "RAW", "D"] = acc_i.events.loc[acc_c.events.Type != "RAW", "D"]

    sum_motp_i = mh.compute(acc_c, metrics=["motp"], name="acc")
    logger.info("MOTP-I = %s", sum_motp_i)
    num_tracks = len(ID_gt_all)

    fn = os.path.basename(path_tracker_output)
    motp_i = sum_motp_i["motp"][0]

    acc_c.events.loc[acc_c.events.Type != "RAW", "D"] = acc_o.events.loc[acc_c.events.Type != "RAW", "D"]
    sum_motp_o = mh.compute(acc_c, metrics=["motp"], name="acc")
    logger.info("MOTP-O = %s", sum_motp_o)
    num_tracks = len(ID_gt_all)

    fn = os.path.basename(path_tracker_output)
    motp_o = sum_motp_o["motp"][0]

    out_string = (
        f"{fn} {num_frames} {mota:.2f} {motp_c:.2f} {motp_o:.2f} {motp_i:.2f} {idf1:.2f} {most_track:.2f} "
        f"{most_lost:.2f} {num_fp} {num_miss} {num_switch} {num_flag} \n"
    )
    out_file.write(out_string)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument(
    #     "--path_tracker_output",
    #     type=str,
    #     default="../../argodataset_30Hz/test_label/028d5cb1-f74d-366c-85ad-84fde69b0fd3",
    # )
    # parser.add_argument(
    #     "--path_labels", type=str, default="../../argodataset_30Hz/labels_v32/028d5cb1-f74d-366c-85ad-84fde69b0fd3"
    # )
    # parser.add_argument("--path_dataset", type=str, default="../../argodataset_30Hz/cvpr_test_set")
    # parser.add_argument("--centroid_method", type=str, default="average", choices=["label_center", "average"])
    # parser.add_argument("--flag", type=str, default="")
    # parser.add_argument("--d_min", type=float, default=0)
    # parser.add_argument("--d_max", type=float, default=100, required=True)

    # args = parser.parse_args()
    # logger.info("args = %s", args)

    log_ids = [
        '7d37fc6b-1028-3f6f-b980-adb5fa73021e',
        '00c561b9-2057-358d-82c6-5b06d76cebcf',
        '1d676737-4110-3f7e-bec0-0c90f74c248f',
        '2d12da1d-5238-3870-bfbc-b281d5e8c1a1',
        '5ab2697b-6e3e-3454-a36a-aba2c6f27818',
        '5f317f5f-3ce9-355b-acf9-386a8c682252',
        '6db21fda-80cd-3f85-b4a7-0aadeb14724d',
        '70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c',
        '85bc130b-97ae-37fb-a129-4fc07c80cca7',
        '033669d3-3d6b-3d3d-bd93-7985d86653ea',
        '33737504-3373-3373-3373-633738571776',
        '39556000-3955-3955-3955-039557148672',
        '64724064-6472-6472-6472-764725145600',
        'aeb73d7a-8257-3225-972e-99307b3a5cb0',
        'b1ca08f1-24b0-3c39-ba4e-d5a92868462c',
        'c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9',
        'cb0cba51-dfaf-34e9-a0c2-d931404c3dd8',
        'cb762bb1-7ce1-3ba5-b53d-13c159b532c8',
        'cd5bb988-092e-396c-8f33-e30969c98535',
        'cd64733a-dd8a-3bdf-b46a-b7144226168a',
        'da734d26-8229-383f-b685-8086e58d1e05',
        'e9a96218-365b-3ecd-a800-ed2c4c306c78',
        'f9fa3960-537f-3151-a1a3-37a9c0d6d7f7',
        'f1008c18-e76e-3c24-adcc-da9858fac145'
    ]

    #exp_name = 'val-track-preds-greedily-matched'
    #tracker_output_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val-track-preds-greedily-matched'
    
    # exp_name = 'val-track-preds-KF-city-frame'
    #tracker_output_dir = f'/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val-track-preds-greedy-match-network-yaw'
    #tracker_output_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val-track-preds-cityframeKF-2dIoU-2019-12-03'


    dataset_dir = '/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val/argoverse-tracking/val'
    d_min = 0
    d_max = 100
    centroid_method = 'average'

    # could do more min hits, but decrease confidence.
    for max_age in range(2,20): # [2,4,5,6,7,8,9,10,20]:
        
        exp_name = f'val-track-preds-KF-city-frame-maxage{max_age}-minhits1-2019-12-04'
        #tracker_output_dir = f'/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/val-track-preds-cityframeKF-2dIoU-2019-12-03-maxage{max_age}-minhits1'
        tracker_output_dir = f'/Users/johnlamb/Downloads/ARGOVERSE-COMPETITION/test-track-preds-cityframeKF-2dIoU-2019-12-04-maxage{max_age}-minhits1'

        out_filename = f"{exp_name}_dmin{int(d_min)}_dmax{int(d_max)}_centroidmethod{centroid_method}.txt"
        logger.info("output file name = %s", out_filename)

        with open(out_filename, "w") as out_file:
            # eval_tracks(
            #     args.path_tracker_output.split(','), args.path_dataset.split(','), args.d_min, args.d_max, out_file, args.centroid_method
            # )

            eval_tracks(
                log_ids,
                tracker_output_dir,
                dataset_dir,
                d_min,
                d_max,
                out_file,
                centroid_method
            )


