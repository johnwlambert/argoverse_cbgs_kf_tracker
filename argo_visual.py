#!/usr/bin/env python2
import sensor_msgs.point_cloud2 as pcl2
import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import json
import sys
from sensor_msgs.msg import PointCloud2, PointField, Image
import numpy as np
from std_msgs.msg import Header
import tf
import os
from plyfile import PlyData, PlyElement
import pandas as pd
import signal

# OpenCV
import cv2
from cv_bridge import CvBridge
cvbridge = CvBridge()


def signal_cb(signum, frame):
    print('Control+C detected.')
    exit()


MARKER_LIFETIME =  0.4

pub = rospy.Publisher('detection', MarkerArray, queue_size=100)
lidar_pub = rospy.Publisher('lidar', PointCloud2, queue_size=100)
ring_front_center_pub = rospy.Publisher('ring_front_center', Image, queue_size=1)

rospy.init_node('visualization_argo', anonymous=True)
#detection result path (The directory until you see lots of log folders)
# folder_path = '/home/samliu/code/argoverse_cbgs_kf_tracker/temp_files/val-split-track-preds-maxage15-minhits5-conf0.3/'
# tracking result path (The directory until you see lots of log folders)
folder_path = '/home/arg/argoverse_cbgs_kf_tracker/temp_files/tracker_output/'
#lidar_data path (The directory until you see lots of log folders)
lidar_path = '/media/arg/arg2TB-Ray/argoverse/test/'
result_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

# Register signal callback
signal.signal(signal.SIGINT, signal_cb)

while not rospy.is_shutdown():
    for dirs in result_dirs:
        ids = []
        tmp_list = [imgfile for imgfile in os.listdir(lidar_path + dirs + '/ring_front_center/')]
        cam_tstamp_list = []
        for imgfile in tmp_list:
            if imgfile.endswith('.jpg'):
                cam_tstamp_list.append(int(imgfile.split('ring_front_center_')[1].split('.jpg')[0]))    

        for filename in sorted(os.listdir(folder_path + dirs + '/per_sweep_annotations_amodal/')):           
            resultsf = open(folder_path + dirs + '/per_sweep_annotations_amodal/' + filename, 'r')
            results = json.load(resultsf)
            count = 0
            tstamp = 0
            markerArray = MarkerArray()

            # Display detection or tracking result
            for result in results:
                marker = Marker()
                marker.header.frame_id='map'
                marker.header.stamp = rospy.Time.from_sec(result['timestamp']/100000000.0)
                marker.ns = 'object'
                marker.id = count
                marker.lifetime = rospy.Duration(MARKER_LIFETIME)#The lifetime of the bounding-box, you can modify it according to the power of your machine.
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.scale.x = result['length']
                marker.scale.y = result['width']
                marker.scale.z = result['height']
                marker.color.b = 1.0
                marker.color.a = 0.5#The alpha of the bounding-box
                marker.pose.position.x = result['center']['x']
                marker.pose.position.y = result['center']['y']
                marker.pose.position.z = result['center']['z']
                marker.pose.orientation.w = result['rotation']['w']
                marker.pose.orientation.x = result['rotation']['x']
                marker.pose.orientation.y = result['rotation']['y']
                marker.pose.orientation.z = result['rotation']['z']
                str_marker = Marker()
                str_marker.header.frame_id='map'
                str_marker.header.stamp = rospy.Time.from_sec(result['timestamp']/100000000.0)
                str_marker.ns = 'text'
                str_marker.id = count
                str_marker.scale.z = 0.8#The size of the text
                str_marker.color.b = 1.0
                str_marker.color.g = 1.0
                str_marker.color.r = 1.0
                str_marker.color.a = 1.0
                str_marker.pose.position.x = result['center']['x']
                str_marker.pose.position.y = result['center']['y']
                str_marker.pose.position.z = result['center']['z']
                str_marker.lifetime = rospy.Duration(MARKER_LIFETIME)
                str_marker.type = Marker.TEXT_VIEW_FACING
                str_marker.action = Marker.ADD
                # str_marker.text = result['label_class'] #for visualize detection

                if result['track_label_uuid'] not in ids: #for visualize tracking---
                    ids.append(result['track_label_uuid'])
                    str_marker.text = str(ids.index(result['track_label_uuid']))
                else:
                    str_marker.text = str(ids.index(result['track_label_uuid']))#----

                markerArray.markers.append(marker)
                markerArray.markers.append(str_marker)
                count=count+1
                tstamp = result['timestamp']

            # Display ground true tracking data if existed 
            if os.path.isdir(lidar_path + dirs + '/per_sweep_annotations_amodal/'):
                resultsf = open(lidar_path + dirs + '/per_sweep_annotations_amodal/' + filename, 'r')
                results = json.load(resultsf)
                for result in results:
                    marker = Marker()
                    marker.header.frame_id='map'
                    marker.header.stamp = rospy.Time.from_sec(result['timestamp']/100000000.0)
                    marker.ns = 'object'
                    marker.id = count
                    marker.lifetime = rospy.Duration(MARKER_LIFETIME)#The lifetime of the bounding-box, you can modify it according to the power of your machine.
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.scale.x = result['length']
                    marker.scale.y = result['width']
                    marker.scale.z = result['height']
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 1.0
                    marker.color.a = 0.2 #The alpha of the bounding-box
                    marker.pose.position.x = result['center']['x']
                    marker.pose.position.y = result['center']['y']
                    marker.pose.position.z = result['center']['z']
                    marker.pose.orientation.w = result['rotation']['w']
                    marker.pose.orientation.x = result['rotation']['x']
                    marker.pose.orientation.y = result['rotation']['y']
                    marker.pose.orientation.z = result['rotation']['z']
                    str_marker = Marker()
                    str_marker.header.frame_id='map'
                    str_marker.header.stamp = rospy.Time.from_sec(result['timestamp']/100000000.0)
                    str_marker.ns = 'text'
                    str_marker.id = count
                    str_marker.scale.z = 0.8#The size of the text
                    str_marker.color.g = 1.0
                    str_marker.color.a = 1.0
                    str_marker.pose.position.x = result['center']['x']
                    str_marker.pose.position.y = result['center']['y']
                    str_marker.pose.position.z = result['center']['z']
                    str_marker.lifetime = rospy.Duration(MARKER_LIFETIME)
                    str_marker.type = Marker.TEXT_VIEW_FACING
                    str_marker.action = Marker.ADD
                    # str_marker.text = result['label_class'] #for visualize detection

                    if result['track_label_uuid'] not in ids: #for visualize tracking---
                        ids.append(result['track_label_uuid'])
                        str_marker.text = str(ids.index(result['track_label_uuid']))
                    else:
                        str_marker.text = str(ids.index(result['track_label_uuid']))#----

                    markerArray.markers.append(marker)
                    markerArray.markers.append(str_marker)
                    count=count+1
                    tstamp = result['timestamp']


            # Display LiDAR data
            for lidar in sorted(os.listdir(lidar_path + dirs + '/lidar/')):
                if str(tstamp) in lidar:
                    try:
                        plydata = PlyData.read(lidar_path + dirs + '/lidar/' + lidar)
                        data = plydata.elements[0].data  
                        data_pd = pd.DataFrame(data)  
                        p = np.zeros(data_pd.shape, dtype=np.float)  
                        property_names = data[0].dtype.names  
                        for i, name in enumerate(property_names):  
                            p[:, i] = data_pd[name]
                    except:
                        continue
                    header = Header()
                    header.stamp = rospy.Time.from_sec(tstamp/100000000.0)
                    header.frame_id = 'map'
                    fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1), PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.UINT32, 1),]
                    lidar_pub.publish(pcl2.create_cloud(header,fields,p[:,:4]))


            # Display Front camera data
            closest_tstamp = min(cam_tstamp_list, key=lambda x:abs(x-tstamp))
            if closest_tstamp != None:
                imgfile = 'ring_front_center_' + str(closest_tstamp) + '.jpg'
                filepath = os.path.join(lidar_path + dirs + '/ring_front_center/', imgfile)
                cv_image = cv2.imread(filepath)
                img_msg = cvbridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
                ring_front_center_pub.publish(img_msg)

            if markerArray:
                print(dirs + '/' + filename)
                pub.publish(markerArray)
            rospy.sleep(0.1)
    break
    