For convenience, we provide precomputed detections for the `training`, `validation`, and `testing` splits.
The file structure of the detections is as follows:

```
- detections
    - training <-- 65 logs
        - 0ef28d5c-ae34-370b-99e7-6709e1c4b929 <-- Log id
            - per_sweep_annotations_amodal <-- for visualization
                - tracked_object_labels_315969338019829000.json <-- Sweep detections
    - validation <-- 24 logs
    - testing <-- 24 logs
```

The structure shown above applies to both validation and testing as well.
The schema of each `.json` file is defined below

```
{
    "center": { <-- Center of the object cuboid, given as x, y, z
        "x": 2.709212303161621,
        "y": 3.548607349395752,
        "z": 0.5272418856620789
    },
    "height": 1.771566390991211, <-- Object height
    "label_class": "VEHICLE", <-- Argmax label from detector
    "length": 4.258421897888184, <-- Object length
    "occlusion": 0, <-- Not used
    "rotation": { <-- Object pose represented as a quaternion
        "w": -0.9999209457658587,
        "x": 0.0,
        "y": 0.0,
        "z": 0.012573870474545222
    },
    "score": 0.9322837591171265, <-- Confidence for the corresponding `label_class`
    "timestamp": 315972349019942000, <-- Corresponding LiDAR sweep timestamp
    "track_label_uuid": null, <-- Not used
    "tracked": true, <-- Not used
    "width": 1.8624107837677002 <-- Object width
},
```

*Note*: Please refer to `object_label_record.py` in `argoverse-api` for loading information and usage.
*Note*: All of the logs found in the four parts of the training set are found in the `training` folder.