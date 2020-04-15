#!/usr/bin/env python3

from numba import jit
import numpy as np

from iou_utils import (
	shapely_polygon_intersection, 
	shapely_polygon_area,
	compute_iou_2d
)


def test_compute_iou_2d():
	"""
	"""
	bbox1 = np.array(
		[
			[0,0],
			[3,0],
			[3,3],
			[0,3]
		])
	bbox2 = np.array(
		[
			[2,1],
			[4,1],
			[4,4],
			[2,4]
		])
	iou_2d = compute_iou_2d(bbox1, bbox2)
	gt_iou_2d = 2 / (9+6-2)
	assert iou_2d == gt_iou_2d


def test_shapely_polygon_intersection1():
	"""
	"""
	poly1 = np.array(
		[
			[0,0],
			[3,0],
			[3,3],
			[0,3]
		])
	poly2 = np.array(
		[
			[2,1],
			[5,1],
			[5,4],
			[2,4]
		])
	inter_area = shapely_polygon_intersection(poly1, poly2)
	assert inter_area == 2
	assert shapely_polygon_area(poly1) == 9
	assert shapely_polygon_area(poly2) == 9


def test_shapely_polygon_intersection2():
	"""
	"""
	poly1 = np.array(
		[
			[0,0],
			[4,0],
			[4,4],
			[0,4]
		])
	poly2 = np.array(
		[
			[0,0],
			[4,0],
			[4,4],
		])
	inter_area = shapely_polygon_intersection(poly1, poly2)
	assert inter_area == 8

	assert shapely_polygon_area(poly1) == 16
	assert shapely_polygon_area(poly2) == 8


if __name__ == '__main__':
	""" """
	test_shapely_polygon_intersection1()
	test_shapely_polygon_intersection2()
	test_compute_iou_2d()
