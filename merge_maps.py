import numpy as np
import datetime
import multiprocessing
from src.merging import *
import gc
import os

def worker(lock, batches):
    while True:
        # Acquire the lock to check and update the directories list
        with lock:
            if not batches:
                break  # No more directories to process

            batch = batches.pop(0)  # Get the next directory
        available_regions=np.array([i for i in range(5)])
        print(np.size(available_regions))
        merge_maps_and_save(pred_dir, pred_file_pattern, batch, output_nc_dir, mask_filename, dist_filename, mdt_filename, network_name, available_regions,L=250e3, crop_pixels=9, dx=7.5e3, with_grads=True, mask_coast_dist=0, lon_min=90 ,lon_max=95, lat_min=0, lat_max=5, res=1/10, progress=False)
        gc.collect()

def create_sublists(large_list, n):
    sublists = [[] for _ in range(n)]

    for i, element in enumerate(large_list):
        sublist_index = i % n
        sublists[sublist_index].append(element)

    return sublists

if __name__ == '__main__':
    pred_dir = './preds_refactored/'
    pred_file_pattern = 'simvp_ssh_sst_ns1000000_global__pred_'
    pred_dates = [datetime.date(2025,1,1)+datetime.timedelta(days=t) for t in range(5)] 
    output_nc_dir = './SimVP_SSH-SST_1M_grads/'
    os.system("mkdir "+output_nc_dir)
    mask_filename = './land_water_mask_10grid.nc' # find in Harvard Dataverse repo
    dist_filename = './distance_to_nearest_coastlines_10grid.nc' # find in Harvard Dataverse repo
    mdt_filename = './mdt_hybrid_cnes_cls18_cmems2020_global.nc' # Chosen MDT, available from AVISO+/CMEMS
    network_name = 'SimVP_SSH_SST_1M_global'
    N_workers = 6 #number of cpus to parallelise across
    
    centers = pred_dates
    
    lock = multiprocessing.Lock()
    num_workers = N_workers
    batches_split = create_sublists(centers, num_workers)
   
    processes = []
    
    for i in range(num_workers):
        worker_batches = batches_split[i]

        process = multiprocessing.Process(target=worker, args=(lock, worker_batches))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()    
