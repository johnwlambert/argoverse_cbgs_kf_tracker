#!/usr/bin/env python3

from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np
import pdb
from sklearn.utils.linear_assignment_ import linear_assignment
import sys
import time

from transform_utils import convert_3dbox_to_8corner
from iou_utils import compute_iou_2d_bboxes


class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, bbox3D, info):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=10, dim_z=7)       
    self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                          [0,1,0,0,0,0,0,0,1,0],
                          [0,0,1,0,0,0,0,0,0,1],
                          [0,0,0,1,0,0,0,0,0,0],  
                          [0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0],
                          [0,0,0,0,0,0,0,1,0,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,0,1]])     
    
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                          [0,1,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0,0,0],
                          [0,0,0,1,0,0,0,0,0,0],
                          [0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0]])

    # with angular velocity
    # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
    # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
    #                       [0,1,0,0,0,0,0,0,1,0,0],
    #                       [0,0,1,0,0,0,0,0,0,1,0],
    #                       [0,0,0,1,0,0,0,0,0,0,1],  
    #                       [0,0,0,0,1,0,0,0,0,0,0],
    #                       [0,0,0,0,0,1,0,0,0,0,0],
    #                       [0,0,0,0,0,0,1,0,0,0,0],
    #                       [0,0,0,0,0,0,0,1,0,0,0],
    #                       [0,0,0,0,0,0,0,0,1,0,0],
    #                       [0,0,0,0,0,0,0,0,0,1,0],
    #                       [0,0,0,0,0,0,0,0,0,0,1]])     
    
    # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
    #                       [0,1,0,0,0,0,0,0,0,0,0],
    #                       [0,0,1,0,0,0,0,0,0,0,0],
    #                       [0,0,0,1,0,0,0,0,0,0,0],
    #                       [0,0,0,0,1,0,0,0,0,0,0],
    #                       [0,0,0,0,0,1,0,0,0,0,0],
    #                       [0,0,0,0,0,0,1,0,0,0,0]])

    # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
    self.kf.P[7:,7:] *= 1000. #state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
    self.kf.P *= 10.
    
    # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
    self.kf.Q[7:,7:] *= 0.01
    self.kf.x[:7] = bbox3D.reshape((7, 1))

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 1           # number of total hits including the first detection
    self.hit_streak = 1     # number of continuing hit considering the first detection
    self.first_continuing_hit = 1
    self.still_first = True
    self.age = 0
    self.info = info        # other info

  def update(self, bbox3D, info): 
    """ 
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1          # number of continuing hit
    if self.still_first:
      self.first_continuing_hit += 1      # number of continuing hit in the fist time
    
    ######################### orientation correction
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    new_theta = bbox3D[3]
    if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
    if new_theta < -np.pi: new_theta += np.pi * 2
    bbox3D[3] = new_theta

    predicted_theta = self.kf.x[3]
    if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
      self.kf.x[3] += np.pi       
      if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
      if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
      
    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
      if new_theta > 0: self.kf.x[3] += np.pi * 2
      else: self.kf.x[3] -= np.pi * 2
    
    ######################### 

    self.kf.update(bbox3D)

    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
    self.info = info

  def predict(self):       
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()      
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.still_first = False
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x[:7].reshape((7, ))




def associate_detections_to_trackers(detections,trackers,iou_threshold=0.1):
# def associate_detections_to_trackers(detections,trackers,iou_threshold=0.01):     # ablation study
# def associate_detections_to_trackers(detections,trackers,iou_threshold=0.25):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  detections:  N x 8 x 3
  trackers:    M x 8 x 3

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)    
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      #print(f'On d={d}, t={t}')
      #iou_matrix[d,t] = iou3d(det,trk)[1] # try 2d iou instead             # det: 8 x 3, trk: 8 x 3
      iou_matrix[d,t] = compute_iou_2d_bboxes(det, trk)

  matched_indices = linear_assignment(-iou_matrix)      # hungarian algorithm

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #print(iou_matrix)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class AB3DMOT(object):
  def __init__(self,max_age=2,min_hits=3):      # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
  # def __init__(self,max_age=3,min_hits=3):        # ablation study
  # def __init__(self,max_age=1,min_hits=3):      
  # def __init__(self,max_age=2,min_hits=1):      
  # def __init__(self,max_age=2,min_hits=5):      
    """              
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    # self.reorder = [3, 4, 5, 6, 2, 1, 0]
    # self.reorder_back = [6, 5, 4, 0, 1, 2, 3]

  def update(self,dets_all):
    """
    Params:
      dets_all: dict
        dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
        info: a array of other info for each det
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
    # dets = dets[:, self.reorder]
    self.frame_count += 1

    trks = np.zeros((len(self.trackers),7))         # N x 7 , #get predicted locations from existing trackers.
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict().reshape((-1, 1))
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]       
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))   
    for t in reversed(to_del):
      self.trackers.pop(t)

    dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
    if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
    else: dets_8corner = []
    trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
    if len(trks_8corner) > 0: trks_8corner = np.stack(trks_8corner, axis=0)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner)
    
    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if t not in unmatched_trks:
        d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
        trk.update(dets[d,:][0], info[d, :][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:        # a scalar of index
        trk = KalmanBoxTracker(dets[i,:], info[i, :]) 
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()      # bbox location
        # d = d[self.reorder_back]

        if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
          ret.append(np.concatenate((d, [trk.id+1], trk.info)).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update >= self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)      # x, y, z, theta, l, w, h, ID, other info, confidence
    return np.empty((0,15))      
    


