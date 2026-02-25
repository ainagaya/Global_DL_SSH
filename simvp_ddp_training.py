import os
import sys
sys.path.append('src')
from src.simvp_model import *
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '55000'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tmp_dir = '~/tmp'
os.environ['TMPDIR'] = tmp_dir
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')

import numpy as np
from src.pytorch_losses import *
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import csv

import xarray as xr

import tacoreader

import requests, io, xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_nc(row):
    """Download a single NetCDF into memory and return (filename, xr.Dataset)."""
    fname = row["id"].split("/")[-1]
    r = requests.get(row["url"], headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return fname, xr.open_dataset(io.BytesIO(r.content), engine="h5netcdf")

def download_all(files_df, max_workers=7):
    """Download all files in parallel into memory, return {filename: xr.Dataset}."""
    datasets = {}
    rows = [row for _, row in files_df.iterrows()]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_nc, row): row for row in rows}
        for f in as_completed(futures):
            fname, ds = f.result()
            if fname in datasets:
                datasets[fname].append(ds)
            else:
                datasets[fname] = [ds]
    return datasets

tacoreader.use("pandas")
dataset = tacoreader.load("https://huggingface.co/datasets/nilsleh/OceanTACO/resolve/main/")

LON_MIN, LON_MAX = 130, 160
LAT_MIN, LAT_MAX =  25,  55

VARS = {
    "glorys.nc" : ("thetao",                            "RdYlBu_r", "SST — GLORYS (°C)",  (5,  28)),
    "l4_sst.nc" : ("analysed_sst",                      "RdYlBu_r", "SST — L4 (°C)",      (5,  28)),
    "l3_sst.nc" : ("adjusted_sea_surface_temperature",  "RdYlBu_r", "SST — L3 (°C)",      (5,  28)),
    "l4_ssh.nc" : ("sla",                               "RdBu_r",   "SSH — L4 (m)",       (-0.5, 0.5)),
    "l3_ssh.nc" : ("sla_filtered",                      "RdBu_r",   "SSH — L3 (m)",       (-0.5, 0.5)),
    "l3_swot.nc": ("ssha_filtered",                     "RdBu_r",   "SSH — SWOT (m)",     (-0.5, 0.5)),
    "l4_sss.nc" : ("sos",                               "viridis",  "SSS — L4 (PSU)",     (32, 38)),
}

#pre-computed global normalisation stats
mean_ssh = 0.074
std_ssh = 0.0986
mean_sst = 293.307
std_sst = 8.726

@tf.function
def normalise_ssh(tensor):
    _mean = mean_ssh
    _std = std_ssh
    
    non_zero_indices = tf.where(tf.not_equal(tensor, 0))
    non_zero_values = tf.gather_nd(tensor, non_zero_indices)
    constant_subtract = _mean
    constant_divide = _std
    subtracted_values = tf.subtract(non_zero_values, constant_subtract)
    updated_values = tf.divide(subtracted_values, constant_divide)
    updated_tensor = tf.tensor_scatter_nd_update(tensor, non_zero_indices, updated_values)
    
    return updated_tensor

@tf.function
def normalise_sst(tensor):
    _mean = mean_sst
    _std = std_sst
    
    non_zero_indices = tf.where(tf.not_equal(tensor, 0))
    non_zero_values = tf.gather_nd(tensor, non_zero_indices)
    constant_subtract = _mean
    constant_divide = _std
    subtracted_values = tf.subtract(non_zero_values, constant_subtract)
    updated_values = tf.divide(subtracted_values, constant_divide)
    updated_tensor = tf.tensor_scatter_nd_update(tensor, non_zero_indices, updated_values)
    
    return updated_tensor

@tf.function
def rescale_x(tensor):
    L_x = 960e3
    n=128
    non_zero_indices = tf.where(tf.not_equal(tensor, 0))
    non_zero_values = tf.gather_nd(tensor, non_zero_indices)
    constant_add = 0.5*L_x
    constant_divide = L_x/(n-1)
    added_values = tf.add(non_zero_values, constant_add)
    updated_values = tf.divide(added_values, constant_divide)
    updated_tensor = tf.tensor_scatter_nd_update(tensor, non_zero_indices, updated_values)
    
    return updated_tensor

@tf.function
def rescale_y(tensor):
    L_y = 960e3
    n=128
    non_zero_indices = tf.where(tf.not_equal(tensor, 0))
    non_zero_values = tf.gather_nd(tensor, non_zero_indices)
    constant_add = 0.5*L_y
    constant_divide = L_y/(n-1)
    added_values = tf.add(-non_zero_values, constant_add)
    updated_values = tf.divide(added_values, constant_divide)
    updated_tensor = tf.tensor_scatter_nd_update(tensor, non_zero_indices, updated_values)
    
    return updated_tensor


class TACO_Dataset(Dataset):
    def __init__(self, taco_dict, split='train', sequence_length=5, n_samples=None):
        """
        Args:
            taco_dict: Dictionary of xarray datasets (one per data source)
            split: 'train' or 'val'
            sequence_length: Number of time steps per sample (default 30)
            n_samples: Number of samples to use (None = use all)
        """
        self.taco_dict = taco_dict
        self.split = split
        self.sequence_length = sequence_length

        print("Downloading files ...")
        print("Files to download:", taco_dict)
        nc_datasets = download_all(taco_dict)
        print(f"\nLoaded {len(nc_datasets)} datasets")

        # Collect all data sources and indices
        self.data_sources = list(nc_datasets.keys())
        self.sample_indices = []
        
        # Build list of valid (source, time_idx) pairs
        merged = {}
        for source in self.data_sources:
            ds = nc_datasets[source]
            ds_merged = []
            if source == "l3_swot.nc" or source == "l3_ssh.nc":
                for daily_dataset in ds:
                # print(f"Processing {source} with dimensions {ds.dims}.")
                    print("Merging tracks and adding time dimension for source:", source)
                    date = daily_dataset.attrs.get("date")
                    print(f"Date for {source}, {daily_dataset}: {date}")
                    daily_dataset_time = self._merge_tracks_add_time(daily_dataset, date)
                    ds_merged.append(daily_dataset_time)
                    print(f"After merging tracks, {source} dimensions: {daily_dataset_time.dims}.")
                # Assuming time dimension is 'time'
                # print("ds.time:", ds.time)
                # merge datasets for each day into single dataset
                merged[source] = xr.concat(ds_merged, dim="time").sortby("time")
                t = merged[source].coords["time"].values
                print("Time coordinate values for source", source, ":", t)
                n_time = merged[source].dims['time']
                # try:
                #     n_time = merged[source].dims['time']
                # except KeyError:
                #     n_time = 1  # If no time dimension, treat as single time step
                for t_idx in range(n_time - sequence_length):
                    self.sample_indices.append((source, t_idx))
            
            if n_samples is not None:
                self.sample_indices = self.sample_indices[:n_samples]

            print(f"Total samples for split '{split}': {len(self.sample_indices)}")

        self.ds = merged
            
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        source, t_idx = self.sample_indices[idx]
        # source_ds = self.ds[source]
        
        # Extract time sequence (adjust variable names as needed)
        # Assuming: 'ssh' = gridded SSH, 'ssh_obs' = observations with lat/lon/value

        # Input: gridded SSH for sequence_length timesteps
        input_data = self.ds['l3_ssh.nc']["sla_filtered"].isel(time=slice(t_idx, t_idx + self.sequence_length)).values
        # print(f"Input data shape for source {source} at time index {t_idx}: {input_data.shape}. Input data: {input_data}")
        # Shape: (sequence_length, lat, lon)
        
        # Output: SWOT-like data
        output_data = self.ds['l3_swot.nc']["ssha_unfiltered"].isel(time=slice(t_idx, t_idx + self.sequence_length)).values
        # print(f"Output data shape for source {source} at time index {t_idx}: {output_data.shape}. Output data: {output_data}")
        # Shape: (sequence_length, n_obs, 3) where 3 = [x, y, value]
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_data).float()
        output_tensor = torch.from_numpy(output_data).float()
        
        # Add channel dimension if needed for input
        if input_tensor.ndim == 3:  # (T, H, W)
            input_tensor = input_tensor.unsqueeze(1)  # (T, 1, H, W)

        if input_tensor.ndim == 4:
           input_tensor = input_tensor.unsqueeze(0)  # (1, T, C, H, W)
        
        return input_tensor, output_tensor

    

    @staticmethod
    def _merge_tracks_add_time(
        ds: xr.Dataset,
        day,
        track_dim: str = "track",
        time_dim: str = "time",
        keep_track_vars: bool = True,
    ) -> xr.Dataset:
        """
        Take a dataset like your L3 grids (lat, lon, track) with per-track metadata vars,
        merge/aggregate all tracks into a single 2D grid for that day, and add a 1-length
        'time' dimension whose value is the given day.

        Rules:
        - For each 2D (lat, lon) variable:
            * If it's integer/bool-ish (e.g. is_overlap): take max across tracks (OR-like).
            * Else: take mean across tracks, skipping NaNs.
        - Non-(lat,lon) vars are kept only if keep_track_vars=True.
        - If there's no track dimension on 2D vars, dataset is treated as already-merged.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset (already opened) for a single date.
        day : str | np.datetime64 | datetime.date | pandas.Timestamp
            The day to stamp into the new time coordinate (e.g. "2023-06-09").
        track_dim : str
            Name of the track dimension (default "track").
        time_dim : str
            Name of the output time dimension (default "time").
        keep_track_vars : bool
                Keep per-track metadata variables (track_ids, track_times, etc.) in output.

        Returns
        -------
        xr.Dataset
            Dataset with variables aggregated over track (if present) and expanded with time dim.
        """
        # Normalize day to datetime64[ns]
        print("day:", day)
        day_str = str(day)  # "20230609"
        iso = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"  # "2023-06-09"
        day64 = np.datetime64(iso, "ns")
        print(day64)  # 2023-06-09T00:00:00.000000000

        out_vars = {}
        for name, da in ds.data_vars.items():
            # Only aggregate variables that are 2D (lat, lon) and also have track_dim
            if track_dim in da.dims and {"lat", "lon"}.issubset(set(da.dims)):
                if np.issubdtype(da.dtype, np.integer) or da.dtype == np.bool_:
                    out_vars[name] = da.max(dim=track_dim, skipna=True)
                else:
                    out_vars[name] = da.mean(dim=track_dim, skipna=True)
            else:
                if keep_track_vars:
                    out_vars[name] = da

        out = xr.Dataset(out_vars, coords={k: v for k, v in ds.coords.items() if k != track_dim})

        # Preserve global attrs
        out.attrs = dict(ds.attrs)

        # Add a time dimension of length 1
        out = out.expand_dims({time_dim: [day64]})

        return out
    
    # ------------------

class LossLoggerCallback:
    def __init__(self, filename):
        self.filename = filename
        self.train_losses = []
        self.val_losses = []

    def __call__(self, epoch, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
            for i in range(len(self.train_losses)):
                writer.writerow([i+1, self.train_losses[i], self.val_losses[i]])
    
# train_dir = './pre-processed/training/'
# val_dir = './pre-processed/validation/'

weight_dir = './model_weights/'
log_dir = './loss_logs/'
viz_dir = './model_preds/'

n_t = 5
L_x = 960e3
L_y = 960e3
n = 128
batch_size = 25 # DON'T CHANGE, THIS IS FIXED IN THE PRE-PROCESSING TO BE 1 BATCH PER FILE
n_obs_max = 400 # max number of SSH observations on any day in loss function, allows to have fixed size inputs/outputs with zero padding making it easier to create TFRecord dataset
n_train_samples = 1000
experiment_name = f'simvp_ssh_sst_ns{n_train_samples}_global_'
num_epochs = 10
workers_per_gpu = 8 # sets the number of CPU processes used per GPU to parallelise the data loading/pre-processing
            
frames = n_t

def train(rank, world_size, checkpoint_path=None):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    lr = 0.001
    n_train_batches = int(n_train_samples/batch_size)
    n_val_batches = 500

    #SSH-SST:
    # model = SimVP_Model_no_skip_sst(in_shape=(n_t,2,128,128),model_type='gsta',hid_S=8,hid_T=128,drop=0.2,drop_path=0.15).to(rank)
    
    #SSH ONLY:
    model = SimVP_Model_no_skip(in_shape=(n_t,1,128,128),model_type='gsta',hid_S=8,hid_T=128,drop=0.2,drop_path=0.15).to(rank)

    # train_files = os.listdir(train_dir)
    # train_dataset_files = [train_dir+f for f in train_files if '.tfrecord' in f]
    # train_dataset_files = train_dataset_files[:n_train_batches]
    # n_train_batches=len(train_dataset_files)

    TRAIN_START_DATE = "2023-05-01"
    TRAIN_END_DATE = "2023-05-15"

    training_files = dataset.sql(f"""
        SELECT
            "l2:id" AS id,
            REPLACE("l2:internal:gdal_vsi", '/vsicurl/', '') AS url
        FROM l2
        WHERE "l0:stac:time_start" >= '{TRAIN_START_DATE}'
        AND "l0:stac:time_start" <  '{TRAIN_END_DATE}'
        AND "l1:id" LIKE '%NORTH_PACIFIC_EAST%'
        """)
    train_dataset = TACO_Dataset(training_files, split="train")

    # val_files = os.listdir(val_dir)
    # val_dataset_files = [val_dir+f for f in val_files if '.tfrecord' in f]
    # val_dataset_files = val_dataset_files[:n_val_batches]

    VAL_START_DATE = "2023-06-01"
    VAL_END_DATE = "2023-06-15"

    val_files = dataset.sql(f"""
        SELECT
            "l2:id" AS id,
            REPLACE("l2:internal:gdal_vsi", '/vsicurl/', '') AS url
        FROM l2
        WHERE "l0:stac:time_start" >= '{VAL_START_DATE}'
        AND "l0:stac:time_start" <  '{VAL_END_DATE}'
        AND "l1:id" LIKE '%NORTH_PACIFIC_EAST%'
        """)
    # val_dataset = TACO_Dataset(val_files, split="val")
    
    # viz_files = os.listdir(val_dir)
    # viz_dataset_files = [val_dir+f for f in viz_files if '.tfrecord' in f]
    # viz_dataset_files = viz_dataset_files[:4]
    VIZ_START_DATE = "2023-07-01"
    VIZ_END_DATE = "2023-07-15"
    viz_files = dataset.sql(f"""
        SELECT
            "l2:id" AS id,
            REPLACE("l2:internal:gdal_vsi", '/vsicurl/', '') AS url
        FROM l2
        WHERE "l0:stac:time_start" >= '{VIZ_START_DATE}'
        AND "l0:stac:time_start" <  '{VIZ_END_DATE}'
        AND "l1:id" LIKE '%NORTH_PACIFIC_EAST%'
        """)
    # viz_dataset = TACO_Dataset(viz_files, split="viz")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # viz_sampler = torch.utils.data.distributed.DistributedSampler(viz_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=workers_per_gpu, sampler=train_sampler)
    # val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=workers_per_gpu, sampler=val_sampler)
    # viz_loader = torch.utils.data.DataLoader(viz_dataset, num_workers=workers_per_gpu, sampler=viz_sampler)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch_tracked_mse_interp
    use_amp =True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_training_steps = checkpoint['scheduler_state_dict']['total_steps']
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,total_steps=total_training_steps)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_logger = LossLoggerCallback(log_dir + experiment_name + f"_startepoch{start_epoch}losses.csv")
    else:
        loss_logger = LossLoggerCallback(log_dir + experiment_name + "_losses.csv")
        start_epoch = 0
        total_training_steps = int(num_epochs*n_train_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,total_steps=total_training_steps)

    for epoch in range(start_epoch, num_epochs):
        #training loop
        model.train()
        train_loss = 0.0
        num_batches=0
        print(f"Epoch {epoch+1}/{num_epochs} - Training...")
        print(f"train_loader", train_loader)
        for torch_input_batch, torch_output_batch in train_loader:
            print(f"torch_input_batch: {torch_input_batch.shape}, torch_output_batch: {torch_output_batch.shape}")
            optimizer.zero_grad(set_to_none=True)
            torch_input_batch = torch_input_batch.squeeze(0).to(rank)
            torch_output_batch = torch_output_batch.squeeze(0).to(rank) 
            print(f"torch_input_batch after sqeeze: {torch_input_batch.shape}, torch_output_batch after sqeeze: {torch_output_batch.shape}")           
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(torch_input_batch)
                loss = criterion(outputs, torch_output_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()
                
            train_loss += loss.item()
            num_batches += 1

        print(train_loss)
        print(num_batches)
        train_loss /= num_batches
        if rank == 0:
            #validation loop
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for torch_input_batch, torch_output_batch in val_loader:
                
                    torch_input_batch = torch_input_batch.squeeze(0).to(rank)
                    torch_output_batch = torch_output_batch.squeeze(0).to(rank)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        val_preds = model(torch_input_batch)
                        val_loss += criterion(val_preds, torch_output_batch).item()
                    num_val_batches += 1

            val_loss /= num_val_batches

            loss_logger(epoch, train_loss, val_loss)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }

            torch.save(checkpoint, weight_dir+experiment_name+f'_weights_epoch{epoch}')
            
            # optional visualisation of predictions during training
            pred = np.zeros((100,30,1,128,128))
            i = 0
            with torch.no_grad():
                for torch_input_batch, torch_output_batch in viz_loader:

                    torch_input_batch = torch_input_batch.squeeze(0).to(rank)
                    torch_output_batch = torch_output_batch.squeeze(0).to(rank)

                    val_preds = model(torch_input_batch)
                    pred[int(25*i):int(25*(i+1)),:,:,:,:] = val_preds.cpu().numpy()
                    i+=1
            np.save(viz_dir+experiment_name+f'pred_viz_epoch{epoch}.npy',pred)

    #clean up
    os.system('rm -r '+tmp_dir+'/*')
    dist.destroy_process_group()


if __name__ == "__main__":
    # num_processes = number of GPUs (currently need to be on same node)
    # num_processes = torch.cuda.device_count()
    num_processes = 1 # set to 1 for debugging, set to number of GPUs for full training
    print(f'Number of GPUs used: {num_processes}')

    mp.spawn(train, args=(num_processes,), nprocs=num_processes,)  # add checkpoint file name here if restarting training
