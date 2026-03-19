# 2023-05-02 Scott Martin
# Code to pre-process the subsetted data into ML-ready input-output pairs, save the pairs in TFRecord chunks of size ~100MB for optimal data pipeline performance.
# stationary gridded variables (bathymetry and MDT) will be appended as additional day at end of time series to allow easy passing to the keras model.

import numpy as np
import datetime
import os
from scipy import stats
import random
import tacoreader
import tensorflow as tf
import time
import multiprocessing

import tacoreader

import requests, io, xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed

# function to list all files within a directory including within any subdirectories
def GetListOfFiles(dirName, ext = '.nc'):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetListOfFiles(fullPath)
        else:
            if fullPath.endswith(ext):
                allFiles.append(fullPath)               
    return allFiles

def serialize_example(input_array, output_array):
        feature = {
            'input': tf.train.Feature(float_list=tf.train.FloatList(value=input_array.flatten())),
            'output': tf.train.Feature(float_list=tf.train.FloatList(value=output_array.flatten()))   
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

def parse_example(serialized_example):
    feature_description = {
        'input': tf.io.FixedLenFeature(int(batch_size*N_t*n*n*2), tf.float32),
        'output': tf.io.FixedLenFeature(int(batch_size*N_t*n_obs_max*3), tf.float32)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)

    input_data = tf.reshape(example['input'], [batch_size, N_t, n, n, 2])
    output_data = tf.reshape(example['output'], [batch_size, N_t, n_obs_max, 3])

    return input_data, output_data

# take available along-track altimetry, randomly select up to n_sats_max sats on each day to use as input, bin average input sats onto zero-padded grid, save output sat(s) un-binned for use in loss function:
def bin_swot(data_tracks, L_x, L_y, n, n_sats_max, filtered=False):
    """
    Extract gridded SSH and raw tracks from xarray Dataset
    
    Args:
        data_tracks: xarray.Dataset containing:
                     - 'ssh' or similar variable with gridded SSH data (128, 128)
                     - 'ssh_tracks' or similar with raw observations
        L_x, L_y: Domain size (not used for pre-gridded data)
        n: Grid size (not used, already 128x128)
        n_sats_max: Not used for single satellite
        filtered: Not used
    
    Returns:
        input_grid: (128, 128) gridded SSH array
        output_tracks: (n_obs, 3) array with [lat, lon, ssh]
    """
    
    # Extract gridded SSH data from xarray
    # TODO cut this into regions?¿
    # input_grid = data_tracks['ssha_filtered'].values  # Convert to numpy

    # Extract coordinates and create meshgrid
    lats_1d = data_tracks.coords['lat'].to_numpy()
    lons_1d = data_tracks.coords['lon'].to_numpy()
    
    # Create 2D meshgrid matching the data shape
    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    
    # Flatten all three to 1D arrays of same length
    lats_flat = lat_grid.flatten()
    lons_flat = lon_grid.flatten()
    values_flat = data_tracks.values.flatten()

    # check if there is any non-nan value in ssh
    if np.isnan(data_tracks.values.flatten()).all():
        print("All SSH values are NaN, returning zero grid and empty tracks.")

     # Now bin with matching-length arrays
     # TODO still 3542x4976 instead of 128x128
    input_grid, _, _, _ = stats.binned_statistic_2d(
        lons_flat, lats_flat, values_flat,
        statistic='mean',
        bins=n,
        # range=[[-L_x/2, L_x/2], [-L_y/2, L_y/2]]
    )
    # input_grid = data_tracks.to_numpy()
    
    # Rotate to correct orientation
    input_grid = np.rot90(input_grid)
    input_grid[np.isnan(input_grid)] = 0
    
    return input_grid

# take available along-track altimetry, randomly select up to n_sats_max sats on each day to use as input, bin average input sats onto zero-padded grid, save output sat(s) un-binned for use in loss function:
def bin_ssh(data_tracks,L_x,L_y, n, n_sats_max, filtered = False):

    _, out_grid = bin_sst(data_tracks, L_x, L_y, n)
    
    return out_grid

    output_tracks = np.stack((lons_flat, lats_flat, values_flat), axis=-1)

    output_tracks[np.isnan(output_tracks)] = 0
def bin_sst(sst_data, L_x, L_y, n):
    """
    Bin lat-lon SST data onto regular grid
    
    Args:
        sst_data: xarray.Dataset with 'sst' variable and lat/lon coordinates
    """

    # quick plot to check data looks correct:
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # sst_plot = sst_data.values
    # plt.imshow(sst_plot, origin='lower')
    # plt.colorbar(label='SST')
    # plt.title('Original SST Data')
    # plt.xlabel('Longitude Index')
    # plt.ylabel('Latitude Index')
    # plt.savefig('original_sst_data.png')
    # plt.close()
    # plt.close()
    # Extract coordinates and create meshgrid
    lats_1d = sst_data.coords['lat'].to_numpy()
    lons_1d = sst_data.coords['lon'].to_numpy()
    
    # Create 2D meshgrid
    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    
    # Flatten all to 1D
    lats_flat = lat_grid.flatten()
    lons_flat = lon_grid.flatten()
    values_flat = sst_data.values.flatten()  # Or whatever SST variable name
    
    # Remove NaN values
    
    # Check if all values are NaN
    if len(values_flat) == 0:
        print("All SST values are NaN, returning zero grid.")
        return np.zeros((n, n)), np.empty((0, 3))
    
    # Bin onto grid
    sst_grid, _, _, _ = stats.binned_statistic_2d(
        lons_flat, lats_flat, values_flat,
        statistic='mean',
        bins=n,
        # range=[[-L_x/2, L_x/2], [-L_y/2, L_y/2]]
    )
    
    # Rotate to correct orientation
    sst_grid = np.rot90(sst_grid)
    sst_grid[np.isnan(sst_grid)] = 0

    # quick plot to check binned data looks correct:
    # plt.figure(figsize=(10, 5))
    # plt.imshow(sst_grid, origin='lower')
    # plt.colorbar(label='Binned SST')
    # plt.title('Binned SST Data')
    # plt.xlabel('Longitude Bin')
    # plt.ylabel('Latitude Bin')
    # plt.savefig('binned_sst_data.png')
    # plt.close()

    output_tracks = np.stack((lons_flat, lats_flat, values_flat), axis=-1)

    output_tracks[np.isnan(output_tracks)] = 0
    
    return sst_grid, output_tracks


batch_size = 25
n_obs_max = 400 # max number of SSH observations on any day in loss function, allows to have fixed size inputs/outputs with zero padding making it easier to create TFRecord dataset
N_t = 30 # length of single input time series in days
n = 128 # no. grid points per side of domain
L_x = 960e3 # size of domain
L_y = 960e3  # size of domain
n_sats_max = 6 # maximum number of altimeters to use in input
filtered = False # whether to use the 65km band-pass filtered or unfiltered SSH observations
test_year = 2025

n_regions = 5

start_date = datetime.date(2024,1,1)
end_date = datetime.date(2025,12,31)
# Define your split (e.g., 60% train, 20% val, 20% test)
# Option 1: Chronological split (realistic for time series)
total_days = (end_date - start_date).days + 1

train_end = start_date + datetime.timedelta(days=int(total_days * 0.6))
val_end = train_end + datetime.timedelta(days=int(total_days * 0.2))

train_dates = []
val_dates = []
test_dates = []

for i in range(total_days):
    current_date = start_date + datetime.timedelta(days=i)
    
    if current_date <= train_end:
        train_dates.append(current_date)
    elif current_date <= val_end:
        val_dates.append(current_date)
    else:
        test_dates.append(current_date)

print(f"Train: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
print(f"Val: {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})")
print(f"Test: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

# Keep 30-day buffer around validation/test to avoid temporal leakage
train_dates_filtered = []
for check_date in train_dates:
    diffs_val = [np.abs((check_date - date).days) for date in val_dates]
    diffs_test = [np.abs((check_date - date).days) for date in test_dates]
    
    if (np.min(diffs_val) >= 30) and (np.min(diffs_test) >= 30):
        train_dates_filtered.append(check_date)

train_dates = train_dates_filtered
# stop if train dates is empty (e.g. if val and test dates are too close together or too close to start/end date):
if train_dates == []:
    raise ValueError('No training dates available, please adjust val/test date ranges or start/end date')

save_regions = True
mode = 'training' # 'validation'
# mode = "validation"
domain = 'global'
if mode == 'training':
    print('Processing training data...')
    save_dir = 'pre-processed/training'
elif mode == 'validation':
    print('Processing validation data...')
    save_dir = 'pre-processed/validation'

# create directories
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if save_regions:
    if not os.path.exists(save_dir+'_regions'):
        os.makedirs(save_dir+'_regions')

regions_available = np.array([r for r in range(n_regions)])

LAT_MIN, LAT_MAX = 0, 5
LON_MIN, LON_MAX = 90, 95

def save_batch(batch):
    batch_no = batch
    filename = save_dir+f'/batch_{batch_no}.tfrecord'

    # WHY this +1???
    input_data_final = np.zeros((batch_size,N_t+1,n,n,2))
    output_npy = np.zeros((batch_size,N_t,n_obs_max,3))
    max_lengths = []
    regions = np.zeros(batch_size,dtype='int')
    for sample in range(batch_size):
        print("Batch", batch, "Sample", sample)
        trying=True
        while trying:
            r = np.random.randint(0,regions_available.shape[0])
            r = regions_available[r]

            lat_center = LAT_MIN + (LAT_MAX - LAT_MIN) * r / n_regions
            lon_center = LON_MIN + (LON_MAX - LON_MIN) * r / n_regions

            if mode=='training':
                available_dates = train_dates
            elif mode=='validation':
                available_dates = val_dates
            mid_date = random.choice(available_dates)

            # files_raw = os.listdir(raw_dir)

            output_data_final = []
            n_tot = 0
            for t_loop in range(N_t):
                date_loop = mid_date - datetime.timedelta(days = N_t/2-t_loop)
                print("Date", date_loop, "time step", t_loop)
                files = dataset.sql(f"""
                    SELECT
                        "l2:id" AS id,
                        REPLACE("l2:internal:gdal_vsi", '/vsicurl/', '') AS url
                    FROM l2
                    WHERE "l0:stac:time_start" LIKE '{date_loop}%'
                    AND "l1:id" LIKE '%NORTH_PACIFIC_EAST%'
                    AND (
                        "l2:id" LIKE '%l3_swot.nc'
                        OR
                        "l2:id" LIKE '%l3_sst.nc'
                        OR
                        "l2:id" LIKE '%l3_ssh.nc'
                    )
                    """)
                nc_datasets = download_all(files)

                try:
                    swot_tracks = nc_datasets["l3_swot.nc"]["ssha_filtered"]
                    ssh_tracks = nc_datasets["l3_ssh.nc"]["sla_filtered"]
                    print("2", swot_tracks.dims)
                except KeyError:
                    print(f"No SWOT data in {date_loop}")
                try:
                    sst_loop = nc_datasets["l3_sst.nc"]["adjusted_sea_surface_temperature"]
                except KeyError:
                    print(f"No SST data in {date_loop}")    

                # cut the data to the region of interest (e.g. 960km x 960km box around center of region):
                lat_min = lat_center - L_y/2/111e3
                lat_max = lat_center + L_y/2/111e3
                lon_min = lon_center - L_x/2/111e3
                lon_max = lon_center + L_x/2/111e3
                
                print(f"Region {r} with center lat {lat_center} and lon {lon_center} has lat range {lat_min} to {lat_max} and lon range {lon_min} to {lon_max}")

                swot_tracks = swot_tracks.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                ssh_tracks = ssh_tracks.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                
                
                if len(swot_tracks)>0:
                    input_ssh = bin_swot(swot_tracks,L_x,L_y, n, n_sats_max, filtered)
                    output_ssh = bin_ssh(ssh_tracks,L_x,L_y, n, n_sats_max, filtered)
                    n_tot+=1
                else:
                    input_ssh = np.zeros((n,n))
                    output_ssh = np.zeros((1,3))

                sst_loop = sst_loop.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                sst_grid, _ = bin_sst(sst_loop, L_x, L_y, n)

                input_data_final[sample,t_loop,:,:,0] = input_ssh
                input_data_final[sample,t_loop,:,:,1] = sst_grid
                output_data_final.append(output_ssh)

            lengths = []
            for i in range(len(output_data_final)):
                lengths.append(output_data_final[i].shape[0])
            for i in range(N_t):
                if lengths[i]<n_obs_max:
                    output_npy[sample,i,:lengths[i],:] = output_data_final[i]
                else:
                    output_npy[sample,i,:,:] = output_data_final[i][:n_obs_max,:]
            sst_total = input_data_final[sample,:,:,:,1]
            # condition to exclude examples with extreme sea ice cover:
            if (np.size(sst_total[sst_total==0])<0.9*np.size(sst_total)) or (n_tot/N_t>=1):
                print("Region", r, "INCLUDED with ", n_tot/N_t, "SSH observations per day on average and", np.size(sst_total[sst_total==0])/np.size(sst_total), "sea ice cover.")
                print(np.sum(n_tot)/N_t, "SSH observations per day on average and")
                print(np.size(sst_total[sst_total==0])/np.size(sst_total), "sea ice cover, ")
                print(f"Being sst_total of empty {np.size(sst_total[sst_total==0])}")
                print(f"Being sst_total {np.size(sst_total)}")
                trying = False
            else:
                print("Region", r, "EXCLUDED with: ")
                print(np.sum(n_tot)/N_t, "SSH observations per day on average and")
                print(np.size(sst_total[sst_total==0])/np.size(sst_total), "sea ice cover, ")
                print(f"Being sst_total of empty {np.size(sst_total[sst_total==0])}")
                print(f"Being sst_total {np.size(sst_total)}")


            regions[sample] = int(r)
            
    if save_regions:
        np.save(save_dir+'_regions'+f'/batch_{batch}_regions.npy',regions)

    

    writer = tf.io.TFRecordWriter(filename)
    serialized_example = serialize_example(input_data_final, output_npy)
    writer.write(serialized_example)

    
def worker(lock, batches, seed):
    np.random.seed(seed)
    while True:
        #acquire lock to check and update the directories list
        with lock:
            if not batches:
                break  

            batch = batches.pop(0)  
            print(f"Worker {multiprocessing.current_process().name} processing batch: {batch}")

        save_batch(batch)

def create_sublists(large_list, n):
    sublists = [[] for _ in range(n)]

    for i, element in enumerate(large_list):
        sublist_index = i % n
        sublists[sublist_index].append(element)

    return sublists

def fetch_nc(row):
    """Download a single NetCDF into memory and return (filename, xr.Dataset)."""
    fname = row["id"].split("/")[-1]
    r = requests.get(row["url"], headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return fname, xr.open_dataset(io.BytesIO(r.content), engine="h5netcdf")

def download_all(files_df, max_workers=1):
    """Download all files in parallel into memory, return {filename: xr.Dataset}."""
    datasets = {}
    rows = [row for _, row in files_df.iterrows()]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_nc, row): row for row in rows}
        for f in as_completed(futures):
            fname, ds = f.result()
            datasets[fname] = ds
    return datasets

if __name__ == '__main__':
    tacoreader.use("pandas")
    dataset = tacoreader.load("https://huggingface.co/datasets/nilsleh/OceanTACO/resolve/main/")
    centers = [i for i in range(4)] # number of batches to process
    
    lock = multiprocessing.Lock()
    num_workers = 1 # number of CPUs to parallelise across
    batches_split = create_sublists(centers, num_workers)
    
    random_seeds = [np.random.randint(0, 100000) for _ in range(num_workers)]
    
    processes = []
    
    for i in range(num_workers):
        worker_batches = batches_split[i]
        random_seed = random_seeds[i]
        
        process = multiprocessing.Process(target=worker, args=(lock, worker_batches, random_seed))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()    

    
    
    
        
        
        
