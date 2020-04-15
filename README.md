# argoverse_cbgs_kf_tracker


## Precomputed 3D Detections
The precomputed 3D detections were computed on the Argoverse dataset using the method described in [Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection](https://arxiv.org/abs/1908.09492), with detection range increased to 100 meters in each direction and pruned to ROI to match Argoverse annotation policy.

The detections can be freely downloaded at our [3d tracking competition page](https://evalai.cloudcv.org/web/challenges/challenge-page/453/overview) [[.zip]](https://s3.amazonaws.com/argoai-argoverse/detections_v1.1b.zip).

## Kalman Filter Tracking
This code extends [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT), subject to its [license](https://github.com/xinshuoweng/AB3DMOT/blob/master/LICENSE). However, instead of tracking in the camera coordinate frame (as AB3DMOT does), we perform tracking in the Argoverse city coordinate frame [(see Argoverse paper and appendix)](https://arxiv.org/abs/1911.02620).

Instead of greedily matching sporadic detections, we solve a number of independent estimation problems (filtering) in a factor graph. Specifically, we use the IoU metric to perform data association (decoupling the estimation problems), and then consider each 3D detection as a measurement of an unknown state for a particular vehicle.

## Results on Argoverse Leaderboard
As of Wednesday April 15, 2020 this implementation took 1st place on the Argoverse 3d tracking test set ([leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/453/leaderboard/1278)). Several per-metric results are here:

   |  Car <br> MOTA  |  Pedestrian <br>MOTA | Car <br> MOTPD   | Pedestrian <br> MOTPD | Car MT <br> (Mostly Tracked) | Pedestrian MT <br> (Mostly Tracked)   | Car <br> FN  | Ped. <br> FN |  
   | :-----: | :----------: | :-----: |  :--------: | :------------------: | :--------------: | :----: | :--:   |
   | 65.90   | 48.31        |  0.34   | 0.37        | 0.51                 | 0.28             | 23,594 | 25,780 |


## Choice of Coordinate Frame

Tracking in the "city frame" is advantageous over tracking in the egovehicle frame or camera coordinate frame since parked cars are constant in the city frame. You can find our technical report [here](https://drive.google.com/file/d/1TlrZDQTz3c9t7lXmUWcatF0sGjv14Era/view?usp=sharing) (runner up at Neurips 19 Argoverse 3D Tracking Competition that used less high-quality detections from PointPillars, achieving 48.33 Car MOTA).

## Running the Code

First, install the `argoverse-api` module from [here](https://github.com/argoai/argoverse-api). Also download the data (egovehicle poses will be necessary),

Next, download the detections [zip file](https://s3.amazonaws.com/argoai-argoverse/detections_v1.1b.zip), unzip them. 

To run the tracker, pass the path to the unzipped detections directory, which should end in `argoverse_detections_2020`, to `run_ab3dmot.py`, as shown below:

```
DETECTIONS_DATAROOT="/path/to/argoverse_detections_2020" # replace with your own path
POSE_DIR="/path/to/argoverse/data" # should be either val or test set directory
SPLIT="val" # should be either 'val' or 'test'
python run_ab3dmot.py --dets_dataroot $DETECTIONS_DATAROOT --pose_dir $POSE_DIR --split $SPLIT
```

<p align="left">
  <img src="videos/de6c96c4-f2b2-3f0f-9971-ed35f4118c1e_ring_front_center_30fps.gif" height="280">
  <img src="videos/21e37598-52d4-345c-8ef9-03ae19615d3d_ring_front_center_30fps.gif" height="280">
</p>
<p align="center">
  <img src="videos/1e5d7745-c7b3-31a0-ae57-c480fcaa220e_ring_front_center_30fps.gif" height="280">
</p>


## Brief Explanation of Repo Contents

- `ab3dmot.py`: kalman filter state management (modified from [original](https://github.com/xinshuoweng/AB3DMOT))
- `detections_README.md`: explanation of how detections are provided
- `iou_utils.py`: simple intersection-over-union utilities
- `run_ab3dmot.py`: execute the tracker on 3d detections which must be provided in egovehicle frame
- `transform_utils.py`: upgrade SE(2) poses to SE(3) and vice versa

- `tests`
    - `test_iou_utils.py`: a few unit tests
- `visualization` (can be ignored): patches on argoverse-api for better visualization/easier eval
    - `cuboids_to_bboxes.py`: improved script for visualizing tracks (original is in argoverse-api)
    - `object_label_record.py`: updated classes to support visualizing tracks (original is in argoverse-api)
    - `eval_tracking.py`: slightly more user-friendly interface for evaluation script
    - `dump_to_argoverse.py`: Lift SE(2) detections (e.g. PointPillars) to SE(3)

## Citing this work
Open-source Implementation
```
@misc{
    author = {John Lambert},
    title = {Open Argoverse CBGS-KF Tracker},
    howpublished={\url{https://github.com/johnwlambert/argoverse_cbgs_kf_tracker}},
    year = {2020},
}
```


## License

This code is provided by myself for purely non-commercial, research purposes. It may not be used commercially in a product without my permission.
