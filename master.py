import pandas as pd
import numpy as np
import json
import gateway_locations
from datetime import datetime, timedelta
import gmplot
import os
import requests
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# import packages
import osmnx as ox
import seaborn as sns

# set seaborn properties
sns.set(font_scale=1.5, style="white")

# load raster packages
from shapely.geometry import Point, Polygon, LineString, box
import xarray as xr
#import rioxarray as rxr
import rasterio as rio
import geopandas as gpd

import earthpy as et
import earthpy.plot as ep
import warnings

from pyproj import Proj
from bokeh.plotting import figure, show, output_file
from bokeh.tile_providers import get_provider, Vendors
from bokeh.io import output_notebook, curdoc, reset_output
from bokeh.palettes import Reds7
from bokeh.models import ColumnDataSource, GeoJSONDataSource, MultiLine, Rect
import contextily as ctx

from matplotlib import pyplot


# Transform SD card data to common DataFrame
def clean_sd(paths, deviceIDs):
    device_ids = range(1, len(paths)+1)
    all_data = []
    for path in paths:
        data = open(path)
        content = data.read()
        content = content.split('\n')
        for i in device_ids:
            if i == int(path.split('/')[-1].split('_')[0].split('sd')[1]):
                all_data.append([x + "," + str(i) for x in content])

    all_data = [x for xs in all_data for x in xs]
    
    output = {
        'deviceID':[], 'lon_tx':[], 'lat_tx':[], 'year':[], 'month':[], 'day':[], 'hour':[], 'min':[], 'sec':[]
    }
    
    for i in all_data:
        if 'LoRaWANSend' in i:
            data = i.split(',')
            output['lat_tx'].append(float(data[0].split(':')[1]))
            output['lon_tx'].append(float(data[1].split(':')[1]))
            output['deviceID'].append(deviceIDs[int(data[11])])
            output['year'].append(int(data[4].split(':')[1]))
            output['month'].append(int(data[5].split(':')[1]))
            output['day'].append(int(data[6].split(':')[1]))
            output['hour'].append(int(data[7].split(':')[1]))
            output['min'].append(int(data[8].split(':')[1]))
            output['sec'].append(int(data[9].split(':')[1].split('}')[0]))
            
    output = pd.DataFrame.from_dict(output)
    
    output['dateInt'] = output['year'].astype(str) + output['month'].astype(str).str.zfill(2) + output['day'].astype(str).str.zfill(2) + output['hour'].astype(str).str.zfill(2) + output['min'].astype(str).str.zfill(2) + output['sec'].astype(str).str.zfill(2)
    dates = output.dateInt.astype(str).values    

    new_dates = []
    for date in dates:
        original_date = datetime.strptime(date, "%Y%m%d%H%M%S")
        timezone_offset = timedelta(hours=5)
        date_with_offset = original_date - timezone_offset
        formatted_date = date_with_offset.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        new_dates.append(formatted_date)

    output['time'] = new_dates
    output = output.drop(['dateInt', 'year', 'month', 'day', 'hour', 'min', 'sec'], axis = 1)
    output = output.reset_index(drop=True)
    return output


# Transform Chronograf database data to common DataFrame
def clean_db(paths, gateway_locations, type):
    # Read CSV files and concatenate into a single DataFrame
    db_data = pd.concat((pd.read_csv(data) for data in paths), ignore_index=True)

    if type == 'glamos':
        db_data = db_data.dropna(subset=['latitude_longitude.latitude', 'latitude_longitude.longitude', 'latitude_longitude.rx_metadata'])
        db_data = db_data.reset_index(drop=True)
        
        temp = db_data['latitude_longitude.rx_metadata'].apply(json.loads)

    else:
        db_data = db_data.dropna(subset=['gps.latitude', 'gps.longitude', 'gps.metadata'])
        db_data = db_data.reset_index(drop=True)
        
        temp = db_data['gps.metadata'].apply(json.loads)

    output = {
        'time':[], 'deviceID':[], 'lon_tx':[], 'lat_tx':[], 'rssi':[], 'snr':[], 'gw_id':[], 'gw_lon':[], 'gw_lat':[]
    }

    for j in range(len(db_data)):
        output['time'].append([0 if 'time' not in i.keys() else i['time'] for i in temp[j]])
        output['rssi'].append([0 if 'rssi' not in i.keys() else i['rssi'] for i in temp[j]])
        output['snr'].append([0 if 'snr' not in i.keys() else i['snr'] for i in temp[j]])
        output['gw_id'].append([i['gateway_ids']['gateway_id'] for i in temp[j]])
        output['gw_lon'].append([gateway_locations[i][1] for i in output['gw_id'][j]])
        output['gw_lat'].append([gateway_locations[i][0] for i in output['gw_id'][j]])

    output['deviceID'].append(db_data['deviceID'].values)
    output['deviceID'] = [[x] for xs in output['deviceID'] for x in xs]

    if type == 'glamos':
        output['lon_tx'].append(db_data['latitude_longitude.longitude'].values)
        output['lon_tx'] = [[x] for xs in output['lon_tx'] for x in xs]
        output['lat_tx'].append(db_data['latitude_longitude.latitude'].values)
        output['lat_tx'] = [[x] for xs in output['lat_tx'] for x in xs]

    else:
        output['lon_tx'].append(db_data['gps.longitude'].values)
        output['lon_tx'] = [[x] for xs in output['lon_tx'] for x in xs]
        output['lat_tx'].append(db_data['gps.latitude'].values)
        output['lat_tx'] = [[x] for xs in output['lat_tx'] for x in xs]

    output = pd.DataFrame.from_dict(output)
    output = output[(output.gw_lon != 0) & (output.lon_tx != 0) & (output.gw_lat != 0) & (output.lat_tx != 0)].reset_index(drop=True)
    return output


# Transform glamos txt data to common DataFrame
def clean_glamos_txt(path, gateway_locations):
    glamos_data = open(path, "r")
    glamos_data = glamos_data.read()
    glamos_data = glamos_data.split(',')
    output = ''
    for i in glamos_data:
        output += i + ','
    output = output.split("b'")
    
    payloads = []
    rx_metadata = []
    for i in output[1:]:
        if 'decoded_payload' in i:
            payload = i.split('decoded_payload":')[1]
            payloads.append(payload.split('rx_metadata":')[0])
            rx_metadata.append(payload.split('rx_metadata":')[1])

    output = {
        'time':[], 'deviceID':[], 'lon_tx':[], 'lat_tx':[], 'rssi':[], 'snr':[], 'gw_id':[], 'gw_lon':[], 'gw_lat':[]
    }
    
    for i in range(len(payloads)):
        output['lat_tx'].append([float(payloads[i].split(',')[3].split(':')[1])])
        output['lon_tx'].append([float(payloads[i].split(',')[4].split(':')[1])])
        output['deviceID'].append(['eui-1d4a7d00005745ad'])
        output['time'].append([j.split(',')[0] for j in rx_metadata[i].split('"time":')][1:])
        output['rssi'].append([float(j.split(',')[0]) for j in rx_metadata[i].split('"rssi":')[1:]])
        output['snr'].append([float(j.split(',')[0]) for j in rx_metadata[i].split('"snr":')[1:]])
        output['gw_id'].append([j.split(',')[0] for j in rx_metadata[i].split('"gateway_id":')[1:]])
        output['gw_lon'].append([gateway_locations[j.strip('"')][1] for j in output['gw_id'][i]])
        output['gw_lat'].append([gateway_locations[j.strip('"')][0] for j in output['gw_id'][i]])

    output = pd.DataFrame.from_dict(output)
    output = output[(output.gw_lon != 0) & (output.lon_tx != 0) & (output.gw_lat != 0) & (output.lat_tx != 0)].reset_index(drop=True)
    return output


def filter_df(df, epsilon, min_samples):
    # Combine transmission longitudes and latitudes into a single array
    transmission_data = df[['lon_tx', 'lat_tx']].values
    transmission_data = np.array([(i[0][0], i[1][0]) for i in transmission_data])
    
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    
    # Fit DBSCAN to transmission data
    dbscan.fit(transmission_data)
    
    # Get cluster labels
    cluster_labels = dbscan.labels_
    
    # Initialize list to store indices of representative points
    representative_point_indices = []
    
    # Iterate over unique cluster labels (excluding noise points labeled as -1)
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label != -1:
            # Get indices of points in current cluster
            cluster_indices = np.where(cluster_labels == label)[0]
            # Choose one point as the representative point (e.g., the first point in the cluster)
            representative_point_index = cluster_indices[0]
            # Add representative point index to list
            representative_point_indices.append(representative_point_index)
    
    # Filter DataFrame based on representative point indices
    df_filtered = df.iloc[representative_point_indices].reset_index(drop=True)

    return df_filtered


def filter_df_fails(df, epsilon, min_samples):
    # Combine transmission longitudes and latitudes into a single array
    transmission_data = df[['lon_tx', 'lat_tx']].values
    transmission_data = np.array([(i[0], i[1]) for i in transmission_data])
    
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    
    # Fit DBSCAN to transmission data
    dbscan.fit(transmission_data)
    
    # Get cluster labels
    cluster_labels = dbscan.labels_
    
    # Initialize list to store indices of representative points
    representative_point_indices = []
    
    # Iterate over unique cluster labels (excluding noise points labeled as -1)
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label != -1:
            # Get indices of points in current cluster
            cluster_indices = np.where(cluster_labels == label)[0]
            # Choose one point as the representative point (e.g., the first point in the cluster)
            representative_point_index = cluster_indices[0]
            # Add representative point index to list
            representative_point_indices.append(representative_point_index)
    
    # Filter DataFrame based on representative point indices
    df_filtered = df.iloc[representative_point_indices].reset_index(drop=True)

    return df_filtered


def clean_unpack_df(df):
    rows = [['time', 'deviceID', 'lon_tx', 'lat_tx', 'rssi', 'snr', 'gw_id', 'gw_lon', 'gw_lat']]
    for i in range(len(df.values)):
        for j in range(len(df.gw_id[i])):
            if len(df.snr[i]) == len(df.rssi[i]):
                if not isinstance(df.time.values[i][j], int):
                    rows.append([
                        df.time.values[i][j].strip('"'),
                        df.deviceID[i][0].strip('"'),
                        df.lon_tx[i][0],
                        df.lat_tx[i][0],
                        df.rssi[i][j],
                        df.snr[i][j],
                        df.gw_id[i][j].strip('"'),
                        df.gw_lon[i][j],
                        df.gw_lat[i][j]
                    ])
                
    db_dataset = pd.DataFrame(rows[1:], columns = rows[0])
    db_dataset['time'] = pd.to_datetime(db_dataset['time'], format="%Y-%m-%dT%H:%M:%S.%f%z")
    db_dataset['time'] = db_dataset['time'].dt.tz_convert('UTC').dt.tz_convert('America/New_York')
    db_dataset['time'] = db_dataset['time'].dt.tz_localize(None).dt.tz_localize('UTC')
    
    times = []
    for t in db_dataset.time:
        times.append(t.strftime('%Y-%m-%dT%H:%M:%S.%f'))

    db_dataset['time'] = times
    db_dataset = db_dataset[(db_dataset.gw_lon != 0) & (db_dataset.lon_tx != 0) & (db_dataset.gw_lat != 0) & (db_dataset.lat_tx != 0)].reset_index(drop=True)
    return db_dataset


# Combine common DataFrames
def combine_datasets(df1, df2):
    df = pd.concat([df1, df2], axis=0).drop_duplicates().reset_index(drop=True)
    return df


# get mercator coordinates
def merc(lat, lon):
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x / lon
    y = (180.0/np.pi) * np.log(np.tan((np.pi/4.0) + lat*(np.pi/180.0)/2.0)) * scale
    return x,y


def extract_attempts(sd, db):
    # output, fail_out = [], [] 
    lats_fail, lons_fail, lats, lons = [], [], [], []
    for i in sd.deviceID.unique():
        temp_db = pd.DataFrame(db[db.deviceID == i].time.unique(), columns = ['time'])
        temp_sd = pd.DataFrame(sd[sd.deviceID == i].time.unique(), columns = ['time'])
    
        # temp_merged = pd.merge(temp_sd, temp_db, on='time', how='outer', indicator=True)
    
        temp_sd_df = sd[sd.deviceID == i]
        temp = db[db.deviceID == i]

        temp_sd_df["time"] = pd.to_datetime(temp_sd_df["time"])
        temp["time"] = pd.to_datetime(temp["time"])

        for k in range(len(temp)):
            lats.append(temp.lat_tx.values[k])
            lons.append(temp.lon_tx.values[k])

        temp_merged = pd.merge_asof(temp_sd_df.reset_index(drop=True).sort_values('time'),
              temp.reset_index(drop=True).sort_values('time'),
              on='time',
              direction = 'nearest',
              tolerance = pd.Timedelta('10s'))

        for j in range(len(temp_merged)):
            if np.isnan(temp_merged.lon_tx_y.values[j]):
                lats_fail.append(temp_merged.lat_tx_x.values[j])
                lons_fail.append(temp_merged.lon_tx_x.values[j])

            # else:
            #     lats.append(temp_merged.lat_tx_y.values[j])
            #     lons.append(temp_merged.lon_tx_y.values[j])

    #     for j in temp_merged[temp_merged._merge == 'both'].time.values:
    #         if j in temp.time.values:
    #             output.append(temp[temp.time == j].values)
    
    #     for k in temp_merged[temp_merged._merge == 'left_only'].time.values:
    #         if k not in temp.time.values:
    #             fail_out.append(temp_sd_df[temp_sd_df.time == k].values)

    # lats, lons = [], []
    # for i in output:
    #     lats.append(i[0][3])
    #     lons.append(i[0][2])
    
    # lats_fail, lons_fail = [], []
    # for i in fail_out:
    #     lats_fail.append(i[0][2])
    #     lons_fail.append(i[0][1])
    return lats, lons, lats_fail, lons_fail


# Plot data on a gmplot map
def gmplot_data(df, gateway_lats, gateway_lons, apikey, filename, marker_size):  
    lats = df.lat_tx.values
    lons = df.lon_tx.values
    
    lat_center = np.median(lats)
    lon_center = np.median(lons)
    
    gmap = gmplot.GoogleMapPlotter(lat_center, lon_center, zoom=11, apikey=apikey)
    
    gmap.scatter(lats, lons, size = marker_size, marker = False)
    gmap.scatter(gateway_lats, gateway_lons, "#0a5e1d", size = 200, marker = False)
    
    gmap.draw("/Users/alfredorodriguez/Desktop/" + filename + ".html")
    return "Transmission data successfully plotted."


# Plot attempted transmissions on a gmplot map
def gmplot_attempts(lats, lons, lats_fail, lons_fail, gateway_lats, gateway_lons, apikey, filename):
    lat_center = np.median(lats)
    lon_center = np.median(lons)
    
    gmap = gmplot.GoogleMapPlotter(lat_center, lon_center, zoom=11, apikey=apikey)

    gmap.scatter(lats_fail, lons_fail, marker = False, color = 'red', size = 10)
    gmap.scatter(lats, lons, marker = False, color = 'blue', size = 10)
    gmap.scatter(gateway_lats, gateway_lons, "#0a5e1d", size = 200, marker = False)
      
    gmap.draw("/Users/alfredorodriguez/Desktop/" + filename + ".html")
    return "Transmission attempts successfully plotted."


def hillshade(array,azimuth,angle_altitude):
    azimuth = 360.0 - azimuth 
    
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azm_rad = azimuth*np.pi/180. #azimuth in radians
    alt_rad = angle_altitude*np.pi/180. #altitude in radians
 
    shaded = np.sin(alt_rad)*np.sin(slope) + np.cos(alt_rad)*np.cos(slope)*np.cos((azm_rad - np.pi/2.) - aspect)
    
    return 255*(shaded + 1)/2


# Plot links using bokeh
def plot_links(dataset, location, crs, dsm):
    # load road and waterway network
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        area = ox.geocode_to_gdf(location)
        
    # set area crs to raster crs
    crs = 'EPSG:' + crs
    area = area.to_crs(crs)

    dsm = rxr.open_rasterio(dsm, masked=True)  # mask NaNs
    dsm = dsm.squeeze()

    p = Proj('EPSG:4269')
    left, bot, right, top = dsm.rio.bounds()
    e, s = p(left, bot, inverse=True)
    w, n = p(right, top, inverse=True)

    output_notebook()

    ns, ss, es, ws = dataset.lat_tx.max(), dataset.lat_tx.min(), dataset.lon_tx.max(), dataset.lon_tx.min()
    
    # get plot range
    max_x, max_y = merc(ns,es)
    min_x, min_y = merc(ss,ws)

    x = [(merc(n,e)[0] + merc(n,w)[0])/2]
    y = [(merc(n,e)[1] + merc(s,e)[1])/2]
    wi = [merc(s,e)[0] - merc(s,w)[0]]
    hi = [merc(n,w)[1] - merc(s,w)[1]]

    tr_data = []
    gw_data = []
    for i in range(len(dataset)):
        tr_data.append(list(merc(dataset.lat_tx.values[i], dataset.lon_tx.values[i])))
        gw_data.append(list(merc(dataset.gw_lat.values[i], dataset.gw_lon.values[i])))

    # longitude is x, latitude is y
    xs = []
    ys = []
    for i in range(len(tr_data)):
        xs.append(np.array([tr_data[i][0], gw_data[i][0]]))
        ys.append(np.array([tr_data[i][1], gw_data[i][1]]))

    df_tx, df_gw = dataset[['lat_tx', 'lon_tx']], dataset[['gw_lat', 'gw_lon']]

    df_tx_new = []
    df_gw_new = []
    for i in range(len(df_tx.values)):
        df_tx_new.append(list(merc(df_tx.values[i][0], df_tx.values[i][1])))
        df_gw_new.append(list(merc(df_gw.values[i][0], df_gw.values[i][1])))
    df_tx_new = pd.DataFrame(df_tx_new, columns = ['tr_lon', 'tr_lat'])
    df_gw_new = pd.DataFrame(df_gw_new, columns = ['gw_lon', 'gw_lat'])

    # declare plot
    plot = figure(x_range=(min_x, max_x), y_range=(min_y, max_y),
                  x_axis_type="mercator", y_axis_type="mercator",
                  title="Successful links and gateways",
                  width=800, height=520) 

    # add background tile
    plot.add_tile(get_provider(Vendors.CARTODBPOSITRON_RETINA))

    # data source
    source_bx = ColumnDataSource(dict(x = x, y = y, w=wi, h=hi))
    source_lk = ColumnDataSource(dict(xs = xs, ys = ys))
    source_tx = ColumnDataSource(df_tx_new)
    source_gw = ColumnDataSource(df_gw_new)

    # bbox plot
    plot.rect(source=source_bx, x='x', y='y', width='w', height='h',
              fill_alpha=0, line_color='mediumblue', line_width=1)

    # packet origin circles
    plot.circle(source=source_tx, x='tr_lon', y='tr_lat', line_color='mediumblue',
                fill_color='mediumblue', size=1, legend_label="Transmission Origin")

    # get lines
    lines = MultiLine(xs="xs", ys="ys", line_color="magenta", line_width=0.1, line_alpha=1)
    plot.add_glyph(source_lk, lines)

    # GW circles
    plot.circle(source=source_gw, x='gw_lon', y='gw_lat', line_color='black',
                fill_color='lightpink', alpha=1, size=5, legend_label = "Gateway")

    show(plot)


# Plot hillshade
def hillshade_plot(dsm_file, lons, lats, lons_fail, lats_fail, gateway_lons, gateway_lats, filename):
    dtm_dataset = rio.open(dsm_file)
    dtm_data = dtm_dataset.read(1)
    hs_data = hillshade(dtm_dataset.read(1),225,45)

    #Overlay transparent hillshade on DTM:
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ext = [dtm_dataset.bounds.left, dtm_dataset.bounds.right, dtm_dataset.bounds.bottom, dtm_dataset.bounds.top]
    
    im1 = plt.imshow(dtm_data,cmap='terrain_r',extent=ext); 
    #cbar = plt.colorbar(); cbar.set_label('Elevation, m',rotation=270,labelpad=20)
    im2 = plt.imshow(hs_data,cmap='Greys',alpha=0.8,extent=ext); #plt.colorbar()
    ax=plt.gca(); ax.ticklabel_format(useOffset=False, style='plain') #do not use scientific notation 
    rotatexlabels = plt.setp(ax.get_xticklabels(),rotation=90) #rotate x tick labels 90 degrees
    # plt.grid('on'); # plt.colorbar(); 
    
    if len(lons_fail) > 0:
        plt.scatter(lons_fail, lats_fail, c='red', s = 1)

    if len(lons) > 0:
        plt.scatter(lons, lats, c='blue', s = 1)
    plt.scatter(gateway_lons, gateway_lats, c = 'yellow', s = 20)
    plt.xlim(-77.035, -76.95)
    plt.ylim(42.827, 42.905)
    plt.title('Geneva Elevation Map')
    plt.savefig(filename + ".png")


# Plot grid of coverage
def plot_coverage_grid(lons, lats, lons_fail, lats_fail, crs, gateway_lons, gateway_lats, filename):
    successes = [1]*len(lats)
    df = pd.DataFrame(list(zip(lons, lats, successes)), columns = ["longitude", 'latitude', 'success'])
    
    # Convert to GeoDataFrame with Points
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:' + crs)

    successes = [0]*len(lats_fail)
    df_2 = pd.DataFrame(list(zip(lons_fail, lats_fail, successes)), columns = ["longitude", 'latitude', 'success'])
    
    # Convert to GeoDataFrame with Points
    geometry = [Point(lon, lat) for lon, lat in zip(df_2['longitude'], df_2['latitude'])]
    gdf_points_2 = gpd.GeoDataFrame(df_2, geometry=geometry, crs='EPSG:32618')

    gdf_transmissions = pd.concat([gdf_points, gdf_points_2], ignore_index=True)
    
    # Calculate the bounding box of your data
    xmin, ymin, xmax, ymax = gdf_transmissions.total_bounds

    xmin = -76.53
    xmax = -76.47
    ymin = 42.42
    ymax = 42.49
    
    # Adjust the grid size based on your study area
    grid_size = 0.001
    
    # Create a regular grid of boxes using numpy
    x, y = np.meshgrid(np.arange(xmin, xmax, grid_size), np.arange(ymin, ymax, grid_size))
    boxes = [box(x, y, x+grid_size, y+grid_size) for x, y in zip(x.flatten(), y.flatten())]
    
    # Create a GeoDataFrame for the grid boxes
    grid_gdf = gpd.GeoDataFrame(geometry=boxes, crs=gdf_transmissions.crs)
    
    # Create a GeoDataFrame with a single polygon representing the specified bounds
    bounds = gpd.GeoSeries([box(xmin, ymin, xmax, ymax)])
    bounds_gdf = gpd.GeoDataFrame(geometry=bounds, crs='EPSG:4326')
    
    # Ensure CRS match and check for valid geometries
    gdf_transmissions = gdf_transmissions.to_crs(grid_gdf.crs)
    gdf_transmissions = gdf_transmissions[gdf_transmissions['geometry'].is_valid]
    
    # Spatial join to associate transmissions with grid boxes
    transmissions_in_grid = gpd.sjoin(gdf_transmissions, grid_gdf, predicate='within')
    
    # Aggregate success and total counts for each grid cell
    grid_aggregated = transmissions_in_grid.groupby('index_right')['success'].agg(['sum', 'count'])
    grid_aggregated['success_ratio'] = grid_aggregated['sum'] / grid_aggregated['count']
    
    # Merge aggregated data back to the grid GeoDataFrame
    grid_gdf = grid_gdf.merge(grid_aggregated, left_index=True, right_index=True, how='left')
    
    # Plot the grid boxes with color based on success ratio
    fig, ax = plt.subplots(figsize=(10, 10))
    bounds_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth = 0)
    ctx.add_basemap(ax, crs=bounds_gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    grid_gdf.boundary.plot(ax=ax, color=None, edgecolor='black', linewidth=0)  # Plot grid boundaries
    grid_gdf.plot(ax=ax, column='success_ratio', cmap='viridis', legend=True, edgecolor='black', linewidth=0.5, alpha = 0.5)  # Color based on success ratio
    plt.scatter(gateway_lons, gateway_lats)
    plt.legend(['', 'Gateways'])
    plt.title("LoRaWAN Network Transmission Success")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    fig.savefig(filename + '.png', format='png', dpi=1000, transparent=False)

    return grid_gdf


def create_success_failure_df(unpacked_data, gateway_locations):
    ids = []
    for i in unpacked_data.gw_id.values:
        for j in i:
            ids.append(j)
    ids = np.unique(ids)
    
    unpacked_data['success'] = [list(ids)]*len(unpacked_data)

    successes = np.zeros((len(unpacked_data.success), len(ids)))
    rssi = np.zeros((len(unpacked_data.success), len(ids)))
    snr = np.zeros((len(unpacked_data.success), len(ids)))

    for l, item in enumerate(unpacked_data.success):
        j = unpacked_data.gw_id[l]
        r = unpacked_data.rssi[l]
        s = unpacked_data.snr[l]
        for k in range(len(j)):
            for idx, i in enumerate(item):
                if j[k] == i:
                    successes[l, idx] = 1
                    rssi[l, idx] = r[k]
                    snr[l, idx] = s[k]
                    
    unpacked_data['link'] = list(successes)
    unpacked_data['rssi'] = list(rssi)
    unpacked_data['snr'] = list(snr)

    gw_lats = [gateway_locations[i][0] for i in ids]
    gw_lons = [gateway_locations[i][1] for i in ids]

    unpacked_data['gw_lat'] = [gw_lats]*len(unpacked_data)
    unpacked_data['gw_lon'] = [gw_lons]*len(unpacked_data)

    rows = [['time', 'deviceID', 'lon_tx', 'lat_tx', 'rssi', 'snr', 'gw_id', 'gw_lon', 'gw_lat', 'success']]
    for i in range(len(unpacked_data.values)):
        for j in range(len(unpacked_data.gw_lat[i])):
            if len(unpacked_data.snr[i]) == len(unpacked_data.rssi[i]):
                rows.append([
                    unpacked_data.time.values[i][0].strip('"'),
                    unpacked_data.deviceID[i][0].strip('"'),
                    unpacked_data.lon_tx[i][0],
                    unpacked_data.lat_tx[i][0],
                    unpacked_data.rssi[i][j],
                    unpacked_data.snr[i][j],
                    unpacked_data.success[i][j].strip('"'),
                    unpacked_data.gw_lon[i][j],
                    unpacked_data.gw_lat[i][j],
                    unpacked_data.link[i][j]
                ])
                  
    db_dataset = pd.DataFrame(rows[1:], columns = rows[0])
    db_dataset['time'] = pd.to_datetime(db_dataset['time'], format="%Y-%m-%dT%H:%M:%S.%f%z")
    db_dataset['time'] = db_dataset['time'].dt.tz_convert('UTC').dt.tz_convert('America/New_York')
    db_dataset['time'] = db_dataset['time'].dt.tz_localize(None).dt.tz_localize('UTC')
    
    times = []
    for t in db_dataset.time:
        times.append(t.strftime('%Y-%m-%dT%H:%M:%S.%f'))

    db_dataset['time'] = times
    db_dataset = db_dataset[(db_dataset.gw_lon != 0) & (db_dataset.lon_tx != 0) & (db_dataset.gw_lat != 0) & (db_dataset.lat_tx != 0)].drop_duplicates().reset_index(drop=True)
    return db_dataset


def create_geojson_df(success_failure_df, dsm, crs):
    r = rio.open(dsm)

    point_tr = [Point(x, y) for x, y in zip(success_failure_df['lon_tx'], success_failure_df['lat_tx'])]
    point_gw = [Point(x, y) for x, y in zip(success_failure_df['gw_lon'], success_failure_df['gw_lat'])]
    
    # zip points into linestrings
    geometry = [LineString(xy) for xy in zip(point_tr, point_gw)]
    
    # make geopandas df
    gdf = gpd.GeoDataFrame(success_failure_df, geometry=geometry)
    
    #gdf = gdf.set_crs(r.crs)
    gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs('EPSG:' + crs)

    tr = success_failure_df[['lon_tx', 'lat_tx']].values
    gw = success_failure_df[['gw_lon', 'gw_lat']].values

    tr_geo = gpd.GeoDataFrame(tr, geometry = point_tr, crs="EPSG:" + crs)
    gw_geo = gpd.GeoDataFrame(gw, geometry = point_gw, crs="EPSG:" + crs)

    tr_coord_list = [(point.x, point.y) for point in tr_geo['geometry']]
    gw_coord_list = [(point.x, point.y) for point in gw_geo['geometry']]

    tr_geo["values"] = [x[0] for x in r.sample(tr_coord_list)]
    gw_geo["values"] = [x[0] for x in r.sample(gw_coord_list)]

    gdf.geometry = gdf.geometry.to_crs("EPSG:" + crs)

    gdf['ele_tr'] = tr_geo['values'].values
    gdf['ele_tr'] = gdf['ele_tr'] + 1
    gdf['ele_gw'] = gw_geo['values'].values
    gdf['dist'] = gdf.geometry.length.values

    gdf = gdf.reset_index(drop=True)

    gdf = gdf.to_crs("EPSG:4269")
    print("Dataframe ready to be converted to GeoJSON.")
    return gdf


def test_project_to_utm(coordinates, source_crs="EPSG:4326"):
    # Create a Point geometry
    geometry = Point(coordinates)

    # Create a GeoDataFrame with the Point geometry
    gdf = gpd.GeoDataFrame(geometry=[geometry], crs=source_crs)

    # Project to UTM
    utm_crs = "EPSG:32618"  # UTM Zone 18N for Geneva, NY
    gdf_utm = gdf.to_crs(utm_crs)

    # Extract the UTM coordinates
    utm_x, utm_y = gdf_utm.geometry.x.values[0], gdf_utm.geometry.y.values[0]

    return utm_x, utm_y


def test_fill_rectangle(center, horizontal_length, vertical_length, point_distance):
    """Fill rectanlge with points"""

    # Extracting center coordinates
    cx, cy = center

    # Calculate the number of points in each dimension
    num_horizontal_points = int(horizontal_length / point_distance)
    num_vertical_points = int(vertical_length / point_distance)

    # Calculate the step size for evenly spaced points
    horizontal_step = horizontal_length / num_horizontal_points
    vertical_step = vertical_length / num_vertical_points

    # Generate evenly spaced points inside the rectangle
    points = []
    for i in range(num_horizontal_points):
        for j in range(num_vertical_points):
            x = cx - horizontal_length / 2 + i * horizontal_step
            y = cy - vertical_length / 2 + j * vertical_step
            points.append((x, y))

    return points


def test_create_geodataframe(coordinates, crs="EPSG:32618"):
    # Create a GeoDataFrame with Point geometries
    geometry = [Point(x, y) for x, y in coordinates]
    gdf = gpd.GeoDataFrame(geometry=geometry, crs=crs)

    return gdf


def test_create_lines_to_gateway(points_gdf, gateway_point):
    # Create a GeoDataFrame for LineString geometries
    lines = []

    for point in points_gdf.geometry:
        line = LineString([point, gateway_point])
        lines.append(line)

    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=points_gdf.crs)

    return lines_gdf


def extract_sd_failures(paths, gateways, deviceIDs, gateway_lons, gateway_lats):
    device_ids = list(deviceIDs.keys())
    all_data = []
    for path in paths:
        data = open(path)
        content = data.read()
        content = content.split('\n')
        for i in device_ids:
            if i == int(path.split('/')[-1].split('_')[0].split('sd')[1]):
                all_data.append([x + "," + str(i) for x in content])
    all_data = [x for xs in all_data for x in xs]
    output = {
        'deviceID':[], 'lon_tx':[], 'lat_tx':[], 'rssi':[], 'snr':[], 'year':[], 'month':[], 'day':[], 'hour':[], 'min':[], 'sec':[], 'success':[]
    }

    for i in all_data:
        if 'LoRaWANSend' in i:
            data = i.split(',')
            output['lat_tx'].append(float(data[0].split(':')[1]))
            output['lon_tx'].append(float(data[1].split(':')[1]))
            output['deviceID'].append(deviceIDs[int(data[-1])])
            output['year'].append(int(data[4].split(':')[1]))
            output['month'].append(int(data[5].split(':')[1]))
            output['day'].append(int(data[6].split(':')[1]))
            output['hour'].append(int(data[7].split(':')[1]))
            output['min'].append(int(data[8].split(':')[1]))
            output['sec'].append(int(data[9].split(':')[1].split('}')[0]))
            output['rssi'].append(0)
            output['snr'].append(0)
            output['success'].append(2)
    
    output = pd.DataFrame.from_dict(output)
    
    output['dateInt'] = output['year'].astype(str) + output['month'].astype(str).str.zfill(2) + output['day'].astype(str).str.zfill(2) + output['hour'].astype(str).str.zfill(2) + output['min'].astype(str).str.zfill(2) + output['sec'].astype(str).str.zfill(2)
    dates = output.dateInt.astype(str).values    
    
    new_dates = []
    for date in dates:
        original_date = datetime.strptime(date, "%Y%m%d%H%M%S")
        timezone_offset = timedelta(hours=5)
        date_with_offset = original_date - timezone_offset
        formatted_date = date_with_offset.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        new_dates.append(formatted_date)
    
    output['time'] = new_dates
    output = output.drop(['dateInt', 'year', 'month', 'day', 'hour', 'min', 'sec'], axis = 1)
    output = output.reset_index(drop=True)
    cols = output.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output = output[cols]

    data = output[output.success == 2].reset_index(drop=True)

    gw_ids = gateways*len(data)
    gw_lons = gateway_lons*len(data)
    gw_lats = gateway_lats*len(data)

    data = pd.DataFrame(np.repeat(data.values, len(gateways), axis=0), columns = [
    "time", "deviceID", "lon_tx", "lat_tx", "rssi", "snr", "success"
    ])

    data["gw_id"] = gw_ids
    data["gw_lon"] = gw_lons
    data["gw_lat"] = gw_lats

    cols = data.columns.tolist()
    cols =  cols[0:6] + cols[7:] + [cols[6]]
    data = data[cols]
    return data


# Transform Chronograf database data to common DataFrame
def clean_db_brooklyn(paths, type):
    # Read CSV files and concatenate into a single DataFrame
    db_data = pd.concat((pd.read_csv(data) for data in paths), ignore_index=True)

    if type == 'glamos':
        db_data = db_data.dropna(subset=['latitude_longitude.latitude', 'latitude_longitude.longitude', 'latitude_longitude.rx_metadata'])
        db_data = db_data.reset_index(drop=True)
        
        temp = db_data['latitude_longitude.rx_metadata'].apply(json.loads)

    else:
        db_data = db_data.dropna(subset=['gps.latitude', 'gps.longitude', 'gps.metadata'])
        db_data = db_data.reset_index(drop=True)
        
        temp = db_data['gps.metadata'].apply(json.loads)

    output = {
        'time':[], 'deviceID':[], 'lon_tx':[], 'lat_tx':[], 'rssi':[], 'snr':[], 'gw_id':[]
    }

    for j in range(len(db_data)):
        output['time'].append(['2024-01-01T01:01:01Z' if 'time' not in i.keys() else i['time'] for i in temp[j]])
        output['rssi'].append([0 if 'rssi' not in i.keys() else i['rssi'] for i in temp[j]])
        output['snr'].append([0 if 'snr' not in i.keys() else i['snr'] for i in temp[j]])
        output['gw_id'].append([i['gateway_ids']['gateway_id'] for i in temp[j]])
        # output['gw_lon'].append([gateway_locations[i][1] for i in output['gw_id'][j]])
        # output['gw_lat'].append([gateway_locations[i][0] for i in output['gw_id'][j]])

    output['deviceID'].append(db_data['deviceID'].values)
    output['deviceID'] = [[x] for xs in output['deviceID'] for x in xs]

    if type == 'glamos':
        output['lon_tx'].append(db_data['latitude_longitude.longitude'].values)
        output['lon_tx'] = [[x] for xs in output['lon_tx'] for x in xs]
        output['lat_tx'].append(db_data['latitude_longitude.latitude'].values)
        output['lat_tx'] = [[x] for xs in output['lat_tx'] for x in xs]

    else:
        output['lon_tx'].append(db_data['gps.longitude'].values)
        output['lon_tx'] = [[x] for xs in output['lon_tx'] for x in xs]
        output['lat_tx'].append(db_data['gps.latitude'].values)
        output['lat_tx'] = [[x] for xs in output['lat_tx'] for x in xs]

    output = pd.DataFrame.from_dict(output)
    output = output[(output.lon_tx != 0) & (output.lat_tx != 0)].reset_index(drop=True)
    return output


def clean_unpack_brooklyn(df):
    rows = [['time', 'deviceID', 'lon_tx', 'lat_tx', 'rssi', 'snr', 'gw_id']]
    for i in range(len(df.values)):
        for j in range(len(df.gw_id[i])):
            if len(df.snr[i]) == len(df.rssi[i]):
                rows.append([
                    df.time.values[i][j].strip('"'),
                    df.deviceID[i][0].strip('"'),
                    df.lon_tx[i][0],
                    df.lat_tx[i][0],
                    df.rssi[i][j],
                    df.snr[i][j],
                    df.gw_id[i][j].strip('"')
                ])
                
    db_dataset = pd.DataFrame(rows[1:], columns = rows[0])
    db_dataset['time'] = pd.to_datetime(db_dataset['time'], format="%Y-%m-%dT%H:%M:%S.%f%z")
    db_dataset['time'] = db_dataset['time'].dt.tz_convert('UTC').dt.tz_convert('America/New_York')
    db_dataset['time'] = db_dataset['time'].dt.tz_localize(None).dt.tz_localize('UTC')
    
    times = []
    for t in db_dataset.time:
        times.append(t.strftime('%Y-%m-%dT%H:%M:%S.%f'))

    db_dataset['time'] = times
    db_dataset = db_dataset[(db_dataset.lon_tx != 0) & (db_dataset.lat_tx != 0)].reset_index(drop=True)
    return db_dataset