from cellpose import models, io
from cellpose.io import *
from collections import defaultdict
import csv
import geopandas
import glob
import imagej
from jpype import JArray, JInt
from math import isnan
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas
from pandas import DataFrame
from pathlib import Path
import random
import scyjava
import seaborn
import shutil

def add_overlays_to_groups(nearest, traced_ids, ij, imp):
    """Randomly color groups of cells.

    This is Jacques' function.  It seeks to put a single-color outline
    on each cell of a group.
    """
    Overlay = scyjava.jimport('ij.gui.Overlay')
    ov = Overlay()
    rm = ij.RoiManager.getRoiManager()
    rm.runCommand("Associated", "true")
    colors = ["black", "blue", "cyan", "green", "magenta",
              "orange", "red", "white", "yellow"]
    cell_indexes = nearest.keys()
    cell_names = traced_ids.keys()
    for cell in cell_names:
        cell_idx = traced_ids[cell]
        random_color = random.choice(colors)
        for cell_vs_time in cell_idx:
            chosen_index = rm.getIndex(cell_vs_time)
            roi = rm.select(chosen_index)
            overlay_command = f"Overlay.addSelection('{random_color}', 5);"
            ij.py.run_macro(overlay_command)


def collapse_z(raw_dataset, output_files, ij, method='sum', verbose=True):
    """Stack multiple z slices for each timepoint.

    If I understand Jacques' explanation of the quantification methods
    correctly, they sometimes (often?) perform better on the
    z-integration of pixels at each timepoint.  This function performs
    that and sends the stacked slices to the output directory and adds
    the filenames to the output_files dictionary.
    """
    ZProjector = scyjava.jimport("ij.plugin.ZProjector")()
    cellpose_slices = list(output_files.keys())
    slice_number = 0
    collapsed_slices = []
    for slice_name in cellpose_slices:
        output_directory = os.path.dirname(output_files[slice_name]['output_txt'])
        collapsed_directory = os.path.dirname(output_directory)
        collapsed_directory = f"{collapsed_directory}/collapsed"
        os.makedirs(collapsed_directory, exist_ok=True)
        output_filename = Path(f"{collapsed_directory}/frame{slice_number}.tif").as_posix()
        output_files[slice_name]['collapsed_file'] = output_filename
        if os.path.exists(output_filename):
            if verbose:
                print(f"Skipping {output_filename}, it already exists.")
        else:
            larger_slice = raw_dataset[:, :, :, :, slice_number]
            imp = ij.py.to_imageplus(larger_slice)
            z_projector_result = ZProjector.run(imp, method)
            ## z_projector_mask = ij.IJ.run(z_projector_result, "Convert to Mask", "method=Otsu background=Light")
            z_collapsed_image = ij.py.from_java(z_projector_result)
            z_collapsed_dataset = ij.py.to_dataset(z_collapsed_image)
            saved = ij.io().save(z_collapsed_dataset, output_filename)
            if verbose:
                print(f"Saving image {output_filename}.")
        slice_number = slice_number + 1
    return output_files


def convert_slices_to_pandas(slices, verbose=False):
    """Dump the cellpose_result slice data to a single df.

    There is no good reason for me to store the data as a series of
    dataframes within a dictionary except I want to get more
    comfortable with python datastructures.  Thus, this function
    should be extraneous, but serves as a way to go from my hash to a
    single df.
    """
    concatenated = pandas.DataFrame()
    slice_keys = list(slices.keys())
    slice_counter = 0
    for k in slice_keys:
        slice_counter = slice_counter + 1
        current_slice = slices[k]
        if verbose:
            print(f"The slice is {k}")
        slice_number = current_slice['slice_number']
        slice_data = current_slice['measurements']
        slice_data['Frame'] = slice_number
        if slice_counter == 1:
            concatenated = slice_data
        else:
            concatenated = pandas.concat([concatenated, slice_data])
    ## This is a little silly, but I couldn't remember that the index attribute
    ## is the numeric rowname for a moment
    ## The reset_index() does what it says on the tine, and changes the 1:19, 1:20, etc
    ## of each individual time Frame to a single range of 1:2000
    concatenated.index = concatenated.reset_index().index
    return concatenated


def create_cellpose_rois(output_files, ij, raw_image, imp, collapsed=False, verbose=True,
                         delete=False, max_frames=3):
    """Read the text cellpose output files, generate ROIs."""
    cellpose_slices = list(output_files.keys())
    data_info = {}
    for element in range(len(raw_image.dims)):
        name = raw_image.dims[element]
        data_info[name] = raw_image.shape[element]
    num_times = data_info['Time'] + 1
    num_channels = data_info['Channel']
    num_z = data_info['Z']

    Overlay = scyjava.jimport('ij.gui.Overlay')
    ov = Overlay()
    rm = ij.RoiManager.getRoiManager()
    rm.runCommand("Associated", "true")
    slice_directory = ''
    print("Starting to iterate over times.")
    for timepoint in range(1, num_times):
        frame_number = timepoint - 1 ## I used 0-indexed for the frames.
        print(f"Going to time: {timepoint}")
        imp.setT(timepoint)
        slice_name = f"frame_{frame_number}"
        input_tif = output_files[slice_name]['input_file']
        slice_directory_name = os.path.basename(os.path.dirname(os.path.dirname(input_tif)))
        input_txt = output_files[slice_name]['output_txt']
        input_mask = output_files[slice_name]['output_mask']
        ## The logic for this was taken from:
        ## https://stackoverflow.com/questions/73849418/is-there-any-way-to-switch-imagej-macro-code-to-python3-code
        txt_fh = open(input_txt, 'r')
        roi_stats = defaultdict(list)
        frame_xcoords = []
        frame_ycoords = []
        coords_length = []
        ## Now get the slice for this timepoint from the raw data
        for line in txt_fh:
            xy = line.rstrip().split(",")
            xy_coords = [int(element) for element in xy if element not in '']
            x_coords = [int(element) for element in xy[::2] if element not in '']
            y_coords = [int(element) for element in xy[1::2] if element not in '']
            xcoords_jint = JArray(JInt)(x_coords)
            ycoords_jint = JArray(JInt)(y_coords)
            polygon_roi_instance = scyjava.jimport('ij.gui.PolygonRoi')
            roi_instance = scyjava.jimport('ij.gui.Roi')
            imported_polygon = polygon_roi_instance(xcoords_jint, ycoords_jint,
                                                    len(x_coords), int(roi_instance.POLYGON))
            imp.setRoi(imported_polygon)
            added = rm.addRoi(imported_polygon)
            roi_count = rm.getCount() ## Get the current number of ROIs, 1 indexed.
            roi_zero_idx = roi_count - 1
            selected = rm.select(roi_zero_idx)
            time_set = imp.setT(timepoint)
            updated = rm.runCommand("Update")
            print(f"Finished time: {timepoint} ROI: {roi_count}")
        txt_fh.close()
    imp.show()
    roi_index = JArray(JInt)(range(0, rm.getCount()))
    rm.setSelectedIndexes(roi_index)
    rm.runCommand('Save', f"{slice_directory_name}.zip")
    if delete:
        rm.runCommand('Delete')
    return output_files


# Relevant options:
# batch_size(increase for more parallelization), channels(two element list of two element
# channels to segment; the first is the segment, second is optional nucleus;
# internal elements are color channels to query, so [[0,0],[2,3]] means do main cells in
# grayscale and a second with cells in blue, nuclei in green.
# channel_axis, z_axis ? invert (T/F flip pixels from b/w I assume),
# normalize(T/F percentile normalize the data), diameter, do_3d,
# anisotropy (rescaling factor for 3d segmentation), net_avg (average models),
# augment ?, tile ?, resample, interp, flow_threshold, cellprob_threshold (interesting),
# min_size (turned off with -1), stitch_threshold ?, rescale ?.
def invoke_cellpose(input_directory, model_file, channels=[[0, 0]], diameter=160,
                    threshold=0.4, do_3D=False, batch_size=64, verbose=True, gpu=False):
    """Invoke cellpose using individual slices.

    This takes the series of slices from separate_slices() and sends
    them to cellpose with a specific model.  The dictionary it returns
    is the primary datastructure for the various functions which follow.
    """

    # Relevant options:
    # model_type(cyto, nuclei, cyto2), net_avg(T/F if load built in networks and average them)
    model = models.CellposeModel(gpu=gpu, pretrained_model=model_file)
    slice_directory = Path(f"{input_directory}/slices").as_posix()
    files = get_image_files(slice_directory, '_masks', look_one_level_down=False)
    needed_imgs = []
    output_masks = []
    output_txts = []
    output_files = defaultdict(dict)
    existing_files = 0
    count = 0
    for one_file in files:
        print(f"Reading {one_file}")
        cp_output_directory = Path(f"{input_directory}/cellpose").as_posix()
        os.makedirs(cp_output_directory, exist_ok=True)
        f_name = os.path.basename(one_file)
        f_name = os.path.splitext(f_name)[0]
        start_mask = Path(f"{slice_directory}/{f_name}_cp_masks.png").as_posix()
        output_mask = Path(f"{cp_output_directory}/{f_name}_cp_masks.png").as_posix()
        start_txt = Path(f"{slice_directory}/{f_name}_cp_outlines.txt").as_posix()
        output_txt = Path(f"{cp_output_directory}/{f_name}_cp_outlines.txt").as_posix()
        print(f"Adding new txt file: {output_txt}")
        output_files[f_name]['input_file'] = one_file
        output_files[f_name]['start_mask'] = start_mask
        output_files[f_name]['output_mask'] = output_mask
        output_files[f_name]['start_txt'] = start_txt
        output_files[f_name]['output_txt'] = output_txt
        output_files[f_name]['exists'] = False
        if (os.path.exists(start_txt) or os.path.exists(output_txt)):
            existing_files = existing_files + 1
            print(f"This file already exists: {start_txt}")
            output_files[f_name]['exists'] = True
        else:
            print(f"This file does not exist: {start_txt}")
            img = imread(one_file)
            needed_imgs.append(img)
        count = count + 1
    nimg = len(needed_imgs)
    if verbose and nimg > 0:
        print(f"Reading {nimg} images, starting cellpose.")
        masks, flows, styles = model.eval(needed_imgs, diameter=diameter,
                                          channels=channels, flow_threshold=threshold,
                                          do_3D=do_3D, batch_size=batch_size)
        saved = io.save_to_png(needed_imgs, masks, flows, files)
    else:
        print("Returning the output files.")
    return output_files


def move_cellpose_output(output_files, verbose=False):
    """Cellpose puts its output files into the cwd, I want them in specific directories."""
    print(f"Moving cellpose outputs to the cellpose output directory.")
    output_filenames = list(output_files.keys())
    for f_name in output_filenames:
        output_file = output_files[f_name]['output_mask']
        if (os.path.exists(output_file)):
            if verbose:
                print("The output already exists.")
        else:
            if verbose:
                print(f"Moving {output_files[f_name]['start_mask']} to {output_file}")
            mask_moved = shutil.move(output_files[f_name]['start_mask'],
                                     output_files[f_name]['output_mask'])
            txt_moved = shutil.move(output_files[f_name]['start_txt'],
                                    output_files[f_name]['output_txt'])


def nearest_cells_over_time(df, max_dist=200.0, max_prop=None, x_column='X',
                            y_column='Y', verbose=True):
    """Trace cells over time

    If I understand Jacques' goals correctly, the tracing of cells
    over time should be a reasonably tractable problem for the various
    geo-statistics tools to handle; their whole purpose is to
    calculate n-dimensional distances.  So, let us pass my df to one
    of them and see what happens!

    Upon completion, we should get an array(dictionary? I forget) of
    arrays where each primary key is the top-level cell ID.  Each
    internal array is the set of IDs from the geopandas dataframe,
    which contains all of the measurements.  Thus, we can easily
    extract the data for individual cells and play with it.
    """
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df[x_column], df[y_column]))

    final_time = gdf.Frame.max()
    pairwise_distances = []
    for start_time in range(0, final_time):
        i = start_time
        j = i + 1
        ti_idx = gdf.Frame == i
        tj_idx = gdf.Frame == j
        if verbose:
            print(f"Getting distances of dfs {i} and {j}.")
        ti = gdf[ti_idx]
        tj = gdf[tj_idx]
        ti_rows = ti.shape[0]
        tj_rows = tj.shape[0]
        for ti_row in range(0, ti_rows):
            ti_element = ti.iloc[[ti_row, ]]
            titj = geopandas.sjoin_nearest(ti_element, tj, distance_col="pairwise_dist",
                                           max_distance=max_dist)
            chosen_closest_dist = titj.pairwise_dist.min()
            if (isnan(chosen_closest_dist)):
                print(f"This element has no neighbor within {max_dist}.")
            else:
                chosen_closest_cell = titj.pairwise_dist == chosen_closest_dist
                chosen_closest_row = titj[chosen_closest_cell]
                pairwise_distances.append(chosen_closest_row)

    paired = pandas.concat(pairwise_distances)
    id_counter = 0
    ## Cell IDs pointing to a list of cells
    traced = {}
    ## Endpoints pointing to the cell IDs
    ends = {}

    traced_ids = {}
    cellids_to_startid = {}
    for i in range(0, final_time):
        query_idx = paired.Frame_left == i
        query = paired[query_idx]
        for row in query.itertuples():
            start_cell = row.Index
            start_cellid = row.cell_id_left
            end_cell = row.index_right
            end_cellid = row.cell_id_right

            ## If the current cell ID maps to a starting point, then
            ## add the new endpoint to the traced_ids with that starting cell.
            ## Then add another key to cellids_to_startid with the new endpoint.
            if start_cellid in cellids_to_startid:
                parent = cellids_to_startid[start_cellid]
                current_id_list = traced_ids[parent]
                current_id_list.append(end_cellid)
                traced_ids[parent] = current_id_list
                cellids_to_startid[end_cellid] = parent
            else:
                ## If there is no current ID mapping, create one.
                parent = start_cellid
                cellids_to_startid[end_cellid] = parent
                cellids_to_startid[parent] = parent
                traced_ids[parent] = [parent, end_cellid]


            if start_cell in ends.keys():
                cell_id = ends[start_cell]
                current_value = traced[cell_id]
                current_value.append(end_cell)
                traced[cell_id] = current_value
                ends[end_cell] = cell_id
            else:
                id_counter = id_counter + 1
                traced[id_counter] = [start_cell, end_cell]
                ends[end_cell] = id_counter
    return traced, traced_ids, paired, pairwise_distances


def separate_slices(input_file, ij, raw_image = None,
                    wanted_x=True, wanted_y=True, wanted_z=1,
    wanted_channel=2, cpus=8, overwrite=False, verbose=True):
    """Slice an image in preparation for cellpose.

    Eventually this should be smart enough to handle arbitrary
    x,y,z,channels,times as well as able to use multiple cpus for
    saving the data.  In its current implementation, it only saves 1
    z, 1 channel for every frame of an image into a series of files in
    its output directory.
    """

    input_base = os.path.basename(input_file)
    input_dir = os.path.dirname(input_file)
    input_name = os.path.splitext(input_base)[0]
    output_directory = Path(f"{input_dir}/outputs/{input_name}_z{wanted_z}").as_posix()
    slice_directory = Path(f"{output_directory}/slices").as_posix()
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(slice_directory, exist_ok=True)
    if (is.None(raw_image)):
        print("Starting to open the input file, this takes a moment.")
        raw_image = ij.io().open(input_file)
    if verbose:
        print(f"Opened input file, writing images to {output_directory}")

    data_info = {}
    for element in range(len(raw_image.dims)):
        name = raw_image.dims[element]
        data_info[name] = raw_image.shape[element]
    if verbose:
        print(
            f"This dataset has dimensions: X:{data_info['X']}",
            f"Y:{data_info['Y']} Z:{data_info['Z']} Time:{data_info['Time']}",
        )

    slices = []
    for timepoint in range(data_info['Time']):
        wanted_slice = raw_image[:, :, wanted_channel, wanted_z, timepoint]
        slice_data = ij.py.to_dataset(wanted_slice)
        output_filename = Path(f"{output_directory}/slices/frame_{timepoint}.tif").as_posix()
        if os.path.exists(output_filename):
            if overwrite:
                print(f"Rewriting {output_filename}")
                os.remove(output_filename)
                saved = ij.io().save(slice_data, output_filename)
            else:
                if verbose:
                    print(f"Skipping {output_filename}, it already exists.")
        else:
            saved = ij.io().save(slice_data, output_filename)
            if verbose:
                print(f"Saving image {input_name}_{timepoint}.")
        slices.append(wanted_slice)
    print(f"Returning the output directory: {output_directory}")
    return raw_dataset, slices, output_directory


## The following is from a mix of a couple of implementations I found:
## https://pyimagej.readthedocs.io/en/latest/Classic-Segmentation.html
## an alternative method may be taken from:
## https://pyimagej.readthedocs.io/en/latest/Classic-Segmentation.html#segmentation-workflow-with-imagej2
## My goal is to pass the ROI regions to this function and create a similar df.
def slices_to_roi_measurements(cellpose_result, ij, raw_image, imp,
                               collapsed=False, verbose=True, view_channel=4,
                               view_z=10, stop_after=None):
    """Read the text cellpose output files, generate ROIs, and measure.

    I think there are better ways of accomplishing this task than
    using ij.IJ.run(); but this seems to work...  Upon completion,
    this function should add a series of dataframes to the
    cellpose_result dictionary which comprise the various metrics from
    ImageJ's measurement function of the ROIs detected by cellpose.
    """

    showPolygonRoi = scyjava.jimport('ij.gui.PolygonRoi')
    Overlay = scyjava.jimport('ij.gui.Overlay')
    Regions = scyjava.jimport('net.imglib2.roi.Regions')
    LabelRegions = scyjava.jimport('net.imglib2.roi.labeling.LabelRegions')
    ZProjector = scyjava.jimport('ij.plugin.ZProjector')()
    ov = Overlay()
    rm = ij.RoiManager.getRoiManager()
    rm.runCommand("Associated", "true")
    imp.resetDisplayRanges()
    ij.py.run_macro('resetMinAndMax();')

    output_dict = cellpose_result
    cellpose_slices = list(cellpose_result.keys())
    slice_number = 0
    roi_index = 0
    for slice_name in cellpose_slices:
        output_dict[slice_name]['slice_number'] = slice_number
        ## I am not sure if time is 0 or 1 indexed.
        timepoint = slice_number + 1
        print(f"Looking at slice: {slice_number} which is time: {timepoint}")
        imp.setT(timepoint)
        if (view_channel):
            imp.setC(view_channel)
        if (view_z):
            imp.setZ(view_z)

        if (stop_after is not None) and (stop_after > timepoint):
            break
        input_tif = ''
        #if collapsed:
        #    input_tif = cellpose_result[slice_name]['collapsed_file']
        #else:
        #    input_tif = cellpose_result[slice_name]['input_file']
        #slice_dataset = ij.io().open(input_tif)
        #slice_data = ij.py.to_imageplus(slice_dataset)

        input_txt = cellpose_result[slice_name]['output_txt']
        input_mask = cellpose_result[slice_name]['output_mask']
        if verbose:
            print(f"Processing cellpose outline: {input_txt}")
            print(f"Measuring: {input_tif}")
        # convert Dataset to ImagePlus
        ## Added by ATB 20230712, maybe incorrect.
        ## The logic for this was taken from:
        ## https://stackoverflow.com/questions/73849418/is-there-any-way-to-switch-imagej-macro-code-to-python3-code

        ## Set up the measurement parameters
        set_string = f'Set Measurements...'
        measure_string = f'area mean min centroid median skewness kurtosis integrated stack redirect=None decimal=3'
        measure_setup = ij.IJ.run(set_string, measure_string)

        txt_fh = open(input_txt, 'r')
        roi_stats = defaultdict(list)
        slice_element = 0
        slice_roi_names = []
        for line in txt_fh:
            xy = line.rstrip().split(",")
            xy_coords = [int(element) for element in xy if element not in '']
            x_coords = [int(element) for element in xy[::2] if element not in '']
            y_coords = [int(element) for element in xy[1::2] if element not in '']
            xcoords_jint = JArray(JInt)(x_coords)
            ycoords_jint = JArray(JInt)(y_coords)
            polygon_roi_instance = scyjava.jimport('ij.gui.PolygonRoi')
            roi_instance = scyjava.jimport('ij.gui.Roi')
            imported_polygon = polygon_roi_instance(
                xcoords_jint, ycoords_jint, len(x_coords), int(roi_instance.POLYGON)
            )
            roi_name = f"t{slice_number}_c{slice_element}"
            slice_roi_names.append(roi_name)
            imp.setRoi(imported_polygon)
            added = rm.addRoi(imported_polygon)
            current_index = rm.getCount() - 1
            current_name = rm.getName(current_index)
            renamed = rm.rename(current_index, roi_name)
            selected = rm.select(current_index)
            imp.setT(timepoint)
            if (view_channel):
                imp.setC(view_channel)
            #if (view_z):
            #    imp.setZ(view_z)
            rm.runCommand("Update")
            slice_element = slice_element + 1
            roi_index = roi_index + 1
            measured = ij.IJ.run(imp, 'Measure', '')
            ## All ROIs for this frame have been measured, send the results back to a dataframe
        txt_fh.close()
        slice_result = ij.ResultsTable.getResultsTable()
        slice_table = ij.convert().convert(
            slice_result, scyjava.jimport('org.scijava.table.Table')
        )
        slice_measurements = ij.py.from_java(slice_table)
        slice_measurements['cell_id'] = slice_roi_names
        output_dict[slice_name]['measurements'] = slice_measurements
        ij.IJ.run('Clear Results')
        imp.show()
        imp.setOverlay(ov)
        slice_number = slice_number + 1
        ## All frames have been measured

    return output_dict


def start_fiji(base=None, mem='-Xmx128g', location='venv/bin/Fiji.app',
               mode='interactive', input_file=None):
    """Start fiji with some default options and return the data structures of interest.

    Depending on context, one might want access to a few different things provided
    by fiji/pyimagej when starting up.  This function attempts to make the process
    of starting the fiji instance and grabbing the ij, raw image data, and imageplus
    data as easy as possible.
    """
    scyjava.config.add_option(mem)
    start_dir = os.getcwd()
    if base:
        start_dir = base
    ij = imagej.init(Path(location), mode=mode)
    ij.getApp().getInfo(True)
    ij.ui().showUI()
    ## Something about this init() function changes the current working directory.
    os.chdir(start_dir)
    ij.getVersion()
    raw_image = None
    imp = None
    if input_file:
        raw_image = ij.io().open(input_file)
        shown = ij.ui().show(raw_image)
        imp = ij.py.to_imageplus(raw_image)
        imp.show()
    return ij, raw_image, imp

def write_cell_measurements(traced_ids, paired,
                            output='cell_measurements.csv'):
    """Write cell-by-cell measurements to a csv file.

    This uses the to_csv() function provided by pandas to write out the various metrics
    produced by imageJ to a file in cell ID order.  The parental ID is first, followed by
    the current cell, the next cell, then all other metrics.
    """
    cell_num = 0
    for parent_cellid in traced_ids:
        cell_group = traced_ids[parent_cellid]
        for child_cell in cell_group:
            data_idx = paired.cell_id_left == child_cell
            new_row = paired[data_idx]
            ## Move the parent, left, and right cell IDs to the beginning of the row.
            new_row.insert(0, 'parent_cell', parent_cellid)
            left_popped = new_row.pop('cell_id_left')
            new_row.insert(1, 'cell_id_left', left_popped)
            right_popped = new_row.pop('cell_id_right')
            new_row.insert(2, 'cell_id_right', right_popped)
            num_hits = sum(data_idx)
            if (num_hits > 0):
                cell_num = cell_num + 1
                if (cell_num == 1):
                    new_row.to_csv(output, mode='w', header=True)
                else:
                    new_row.to_csv(output, mode='a', header=False)
    return cell_num


def write_nearest_cellids(nearest, output='nearest.csv'):
    """Write a csv file of cell IDs.

    This uses the csv library to write out the cell IDs, one per line.
    """
    with open(output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        field_names = ['parent_cell_id', 'child_cell_ids']
        writer.writerow(field_names)
        for near in nearest.keys():
            value = nearest[near]
            writer.writerow([near, value])
    print(f"Wrote numeric cell IDs to the file: {output}.")
