#!/usr/bin/env python

import geopandas
from math import isnan
import pandas

measurements = pandas.read_csv('bacteria_tracks/cell_measurements.csv')
tracks = pandas.read_csv('bacteria_tracks/exporttracks-1.csv')

measure_gdf = geopandas.GeoDataFrame(
    measurements,
    geometry=geopandas.points_from_xy(measurements['X'], measurements['Y'], measurements['Slice']))

track_gdf = geopandas.GeoDataFrame(
    tracks,
    geometry=geopandas.points_from_xy(tracks['POSITION_X'], tracks['POSITION_Y'], tracks['POSITION_Z']))



ti = measure_gdf
tj = track_gdf
ti_rows = ti.shape[0]
tj_rows = tj.shape[0]
max_dist = 100
pairwise_bacteria = []
for ti_row in range(0, ti_rows):
    print(f"On row: {ti_row}")
    ti_element = ti.iloc[[ti_row, ]]
    titj = geopandas.sjoin_nearest(ti, tj, distance_col="pairwise_dist",
                                   max_distance=max_dist)
    chosen_closest_dist = titj.pairwise_dist.min()
    if (isnan(chosen_closest_dist)):
        print(f"This element has no neighbor within {max_dist}.")
    else:
        chosen_closest_cell = titj.pairwise_dist == chosen_closest_dist
        chosen_closest_row = titj[chosen_closest_cell]
        pairwise_bacteria.append(chosen_closest_row)
