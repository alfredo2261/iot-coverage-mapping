from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd
import torch.nn as nn
import torch
import numpy as np
import proppy.raster as prs
from functools import partial
from multiprocessing import Pool
from time import perf_counter
import rasterio as rio
import pickle as pkl

class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_inputs = 151, n_outputs = 1)
model.load_state_dict(torch.load("ml_models/geneva_lr_jan29.pth"))

def load_and_process(links, path_raster, relative_buffer=0.5, ncols=100, nrows=20):
    links = links.reset_index(drop=True)

    # Link feautres
    lengths = links["distance"].values
    lines = links.geometry
    zheads, ztails = links["ele_tr"].values, links["ele_gw"].values

    # Compute rectangle features
    buffers = lengths * relative_buffer
    angles = prs.angle(lines)
    rects = prs.makeParallelRectangles(lines.values, angles, buffers)
    consts, slopes = prs.getSlope(lines, angles, zheads, ztails)

    # Form Rectangle-objects
    rectangles = []
    for i, _ in enumerate(rects):
        R = prs.Rectangle(i, rects[i], angles[i], consts[i], slopes[i], lengths[i])
        rectangles.append(R)

    # Main "loop" using multiprocessing
    raster = rio.open(path_raster)
    fun_kwargs = {"nrows": nrows, "ncols": ncols, "rasterfile": raster}

    t_start = perf_counter()  # time run

    out = []
    try:
        last_update = t_start
        for prog, rec in enumerate(rectangles):
            now = perf_counter()
            if prog % 1024 == 0:
                print(f"Progress: {prog} / {len(rectangles)}; {str(((now - last_update)*(len(rectangles) - prog))/(60*60))[:5]}hr {' '*32}\r", end="")
            last_update = now
            out.append(prs.normalizedRasterValuesFile(rec, nrows=nrows, ncols=ncols, rasterfile=raster))
    #     with Pool() as p:
    #         out += p.map(partial(prs.normalizedRasterValues, **fun_kwargs), rectangles)
    except KeyboardInterrupt:
        pass
    print("")
    t_stop = perf_counter()
    duration = t_stop - t_start

    return out, duration

def ml_links(links, dems, facs, rng, heightmap):
    old_crs = facs.crs
    latlong_crs = "EPSG:4326"
    dems = dems.to_crs(latlong_crs)
    facs = facs.to_crs(latlong_crs)
    points = gpd.GeoDataFrame(geometry=dems.geometry, crs=dems.crs)

    lines = []
    point_lons = []
    point_lats = []
    gw_lons = []
    gw_lats = []
    fac_alts = []
    point_alts = []
    for gateway_point, fac_ele in zip(facs.geometry, facs['altitude']):
        gw_lon = gateway_point.x
        gw_lat = gateway_point.y
        for point, point_ele in zip(points.geometry, dems['altitude']):
            line = LineString([point, Point(*gateway_point.xy)])
            lines.append(line)
            point_lons.append(point.x)
            point_lats.append(point.y)
            gw_lons.append(gw_lon)
            gw_lats.append(gw_lat)
            fac_alts.append(fac_ele)
            point_alts.append(point_ele)

    links = gpd.GeoDataFrame(geometry=lines, crs=points.crs)

    links['lon_tx'] = np.array(point_lons)
    links['lat_tx'] = np.array(point_lats)

    links['gw_lon'] = np.array(gw_lons)
    links['gw_lat'] = np.array(gw_lat)

    links['ele_tr'] = np.array(point_alts)
    links['ele_gw'] = np.array(fac_alts)

    links['dist'] = links.to_crs(old_crs).geometry.length.values
    links['distance'] = links['dist']

    start_points = []
    for line in links.geometry:
        start_point = Point(line.xy[0][0], line.xy[1][0])
        start_points.append(start_point)
    transmitters = gpd.GeoDataFrame(geometry=start_points, crs=links.crs)

    links['x_tx'] = transmitters.geometry.x
    links['y_tx'] = transmitters.geometry.y

    links['elevation'] = links['ele_tr']

    lengths = links["distance"].values
    lines = links.geometry
    zheads, ztails = links["ele_tr"].values, links["ele_gw"].values

    # links = links.to_crs(old_crs)

    made_data, counter = load_and_process(links, heightmap)
    print(made_data)
    print(counter)

    with open("geneva_links.pkl", "wb") as f:
        pkl.dump(made_data, f)
