from cellpose import models, io
from cellpose.io import *
from collections import defaultdict
import geopandas
import glob
import imagej
from jpype import JArray, JInt
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas
from pandas import DataFrame
from pathlib import Path
import scyjava
import seaborn
import shutil

scyjava.config.add_option('-Xmx64g')
start_dir = os.getcwd()
ij = imagej.init(Path('venv/bin/Fiji.app'), mode='interactive')
ij.getApp().getInfo(True)
ij.ui().showUI()
## Something about this init() function changes the current working directory.
os.chdir(start_dir)
ij.getVersion()

showPolygonRoi = scyjava.jimport('ij.gui.PolygonRoi')
Overlay = scyjava.jimport('ij.gui.Overlay')
Regions = scyjava.jimport('net.imglib2.roi.Regions')
LabelRegions = scyjava.jimport('net.imglib2.roi.labeling.LabelRegions')
ZProjector = scyjava.jimport('ij.plugin.ZProjector')()
ov = Overlay()
