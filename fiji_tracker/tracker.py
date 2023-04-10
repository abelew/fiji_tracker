def separate_slices(
    input_file,
    wanted_x=True,
    wanted_y=True,
    wanted_z=1,
    wanted_channel=2,
    cpus=8,
    overwrite=False,
):
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
    os.makedirs(output_directory, exist_ok=True)
    if verbose:
        print("Starting to open the input file, this takes a moment.")
    raw_dataset = ij.io().open(input_file)
    if verbose:
        print(f"Opened input file, writing images to {output_directory}")

    data_info = {}
    for element in range(len(raw_dataset.dims)):
        name = raw_dataset.dims[element]
        data_info[name] = raw_dataset.shape[element]
    if verbose:
        print(
            f"This dataset has dimensions: X:{data_info['X']}",
            f"Y:{data_info['Y']} Z:{data_info['Z']} Time:{data_info['Time']}",
        )

    slices = []
    for timepoint in range(data_info['Time']):
        wanted_slice = raw_dataset[:, :, wanted_channel, wanted_z, timepoint]
        slice_data = ij.py.to_dataset(wanted_slice)
        output_filename = Path(f"{output_directory}/frame_{timepoint}.tif").as_posix()
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

    return raw_dataset, slices, output_directory


## Relevant options:
## batch_size(increase for more parallelization), channels(two element list of two element
## channels to segment; the first is the segment, second is optional nucleus;
## internal elements are color channels to query, so [[0,0],[2,3]] means do main cells in
## grayscale and a second with cells in blue, nuclei in green.
## channel_axis, z_axis ? invert (T/F flip pixels from b/w I assume),
## normalize(T/F percentile normalize the data), diameter, do_3d,
## anisotropy (rescaling factor for 3d segmentation), net_avg (average models),
## augment ?, tile ?, resample, interp, flow_threshold, cellprob_threshold (interesting),
## min_size (turned off with -1), stitch_threshold ?, rescale ?.
def invoke_cellpose(
    input_directory,
    model_file,
    channels=[[0, 0]],
    diameter=160,
    threshold=0.4,
    do_3D=False,
    batch_size=64,
    verbose=True,
):
    """Invoke cellpose using individual slices.

    This takes the series of slices from separate_slices() and sends
    them to cellpose with a specific model.  The dictionary it returns
    is the primary datastructure for the various functions which follow.
    """

    ## Relevant options:
    ## model_type(cyto, nuclei, cyto2), net_avg(T/F if load built in networks and average them)
    model = models.CellposeModel(pretrained_model=model_file)
    files = get_image_files(input_directory, '_masks', look_one_level_down=False)
    imgs = []
    output_masks = []
    output_txts = []
    output_files = defaultdict(dict)
    existing_files = 0
    count = 0
    for one_file in files:
        print(f"Reading {one_file}")
        cp_output_directory = os.path.dirname(input_directory)
        cp_output_directory = Path(f"{cp_output_directory}/cellpose").as_posix()
        f_name = os.path.basename(one_file)
        f_name = os.path.splitext(f_name)[0]
        output_mask = Path(f"{cp_output_directory}/{f_name}_cp_masks.png").as_posix()
        output_txt = Path(f"{cp_output_directory}/{f_name}_cp_outlines.txt").as_posix()
        print(f"Adding new txt file: {output_txt}")
        output_files[f_name]['input_file'] = one_file
        output_files[f_name]['output_mask'] = output_mask
        output_files[f_name]['output_txt'] = output_txt
        output_files[f_name]['exists'] = False
        if os.path.exists(output_txt):
            existing_files = existing_files + 1
            output_files[f_name]['exists'] = True
        else:
            img = imread(f)
            imgs.append(img)
        if count == 0:
            os.makedirs(cp_output_directory, exist_ok=True)
        count = count + 1
    nimg = len(imgs)
    if verbose and nimg > 0:
        print(f"Read {nimg} images, starting cellpose.")
        masks, flows, styles = model.eval(
            imgs,
            diameter=diameter,
            channels=channels,
            flow_threshold=threshold,
            do_3D=do_3D,
            batch_size=batch_size,
        )
        io.save_to_png(imgs, masks, flows, files)
    else:
        print("Returning the output files.")
    return output_files


def collapse_z(raw_dataset, cellpose_result, method='sum'):
    """Stack multiple z slices for each timepoint.

    If I understand Jacques' explanation of the quantification methods
    correctly, they sometimes (often?) perform better on the
    z-integration of pixels at each timepoint.  This function performs
    that and sends the stacked slices to the output directory and adds
    the filenames to the cellpose_result dictionary.
    """
    cellpose_slices = list(cellpose_result.keys())
    slice_number = 0
    collapsed_slices = []
    for slice_name in cellpose_slices:
        output_directory = os.path.dirname(cellpose_result[slice_name]['output_txt'])
        collapsed_directory = os.path.dirname(output_directory)
        collapsed_directory = f"{collapsed_directory}/collapsed"
        os.makedirs(collapsed_directory, exist_ok=True)
        output_filename = Path(
            f"{collapsed_directory}/frame{slice_number}.tif"
        ).as_posix()
        cellpose_result[slice_name]['collapsed_file'] = output_filename
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
    return cellpose_result


## The following is from a mix of a couple of implementations I found:
## https://pyimagej.readthedocs.io/en/latest/Classic-Segmentation.html
## an alternative method may be taken from:
## https://pyimagej.readthedocs.io/en/latest/Classic-Segmentation.html#segmentation-workflow-with-imagej2
## My goal is to pass the ROI regions to this function and create a similar df.
def slices_to_roi_measurements(cellpose_result, collapsed=False):
    """Read the text cellpose output files, generate ROIs, and measure.

    I think there are better ways of accomplishing this task than
    using ij.IJ.run(); but this seems to work...  Upon completion,
    this function should add a series of dataframes to the
    cellpose_result dictionary which comprise the various metrics from
    ImageJ's measurement function of the ROIs detected by cellpose.
    """
    output_dict = cellpose_result
    cellpose_slices = list(cellpose_result.keys())
    slice_number = 0
    for slice_name in cellpose_slices:
        output_dict[slice_name]['slice_number'] = slice_number
        input_tif = ''
        if collapsed:
            input_tif = cellpose_result[slice_name]['collapsed_file']
        else:
            input_tif = cellpose_result[slice_name]['input_file']
        slice_dataset = ij.io().open(input_tif)
        slice_data = ij.py.to_imageplus(slice_dataset)
        input_txt = cellpose_result[slice_name]['output_txt']
        input_mask = cellpose_result[slice_name]['output_mask']
        if verbose:
            print(f"Processing cellpose outline: {input_txt}")
            print(f"Measuring: {input_tif}")
        # convert Dataset to ImagePlus
        imp = ij.py.to_imageplus(slice_data)
        rm = ij.RoiManager.getRoiManager()
        rm.runCommand("Associated", "true")
        rm.runCommand("show All with labels")
        ## The logic for this was taken from:
        ## https://stackoverflow.com/questions/73849418/is-there-any-way-to-switch-imagej-macro-code-to-python3-code
        txt_fh = open(input_txt, 'r')
        set_string = f'Set Measurements...'
        measure_string = f'area mean min centroid median skewness kurtosis integrated stack redirect=None decimal=3'
        ij.IJ.run(set_string, measure_string)
        roi_stats = defaultdict(list)
        for line in txt_fh:
            xy = line.rstrip().split(",")
            xy_coords = [int(element) for element in xy]
            x_coords = [int(element) for element in xy[::2]]
            y_coords = [int(element) for element in xy[1::2]]
            xcoords_jint = JArray(JInt)(x_coords)
            ycoords_jint = JArray(JInt)(y_coords)
            polygon_roi_instance = scyjava.jimport('ij.gui.PolygonRoi')
            roi_instance = scyjava.jimport('ij.gui.Roi')
            imported_polygon = polygon_roi_instance(
                xcoords_jint, ycoords_jint, len(x_coords), int(roi_instance.POLYGON)
            )
            imp.setRoi(imported_polygon)
            rm.addRoi(imported_polygon)
            ij.IJ.run(imp, 'Measure', '')
        slice_result = ij.ResultsTable.getResultsTable()
        slice_table = ij.convert().convert(
            slice_result, scyjava.jimport('org.scijava.table.Table')
        )
        slice_measurements = ij.py.from_java(slice_table)
        output_dict[slice_name]['measurements'] = slice_measurements
        ij.IJ.run('Clear Results')
        txt_fh.close()
        imp.setOverlay(ov)
        imp.getProcessor().resetMinAndMax()
        slice_number = slice_number + 1
    return output_dict


def convert_slices_to_pandas(slices):
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


def nearest_cells_over_time(
    df, max_dist=10.0, max_prop=0.7, x_column='X', y_column='Y', verbose=True
):
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
        df, geometry=geopandas.points_from_xy(df[x_column], df[y_column])
    )

    final_time = gdf.Frame.max()
    pairwise_distances = []
    for start_time in range(1, final_time):
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
        titj = geopandas.sjoin_nearest(ti, tj, distance_col="pairwise_dist")
        pairwise_distances.append(titj)

    id_counter = 0
    ## Cell IDs pointing to a list of cells
    traced = {}
    ## Endpoints pointing to the cell IDs
    ends = {}
    for i in range(0, final_time - 1):
        query = pairwise_distances[i]
        passed_idx = query.pairwise_dist <= max_dist
        failed_idx = query.pairwise_dist > max_dist
        if failed_idx.sum() > 0:
            if verbose:
                print(f"Skipped {failed_idx.sum()} elements in segment {i}.")
        query = query[passed_idx]

        prop_change = query.Area_left / query.Area_right
        increased_idx = prop_change > 1.0
        prop_change[increased_idx] = 1.0 / prop_change[increased_idx]
        failed_idx = prop_change < max_prop
        passed_idx = prop_change >= max_prop
        if failed_idx.sum() > 0:
            if verbose:
                skip_string = (
                    f"Skipped {failed_idx.sum()} elements in segment {i} ",
                    f"because the size changed too much.",
                )
                print(skip_string)
            query = query[passed_idx]

        for row in query.itertuples():
            start_cell = row.Index
            end_cell = row.index_right
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
    return traced
