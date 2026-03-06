# train_ddpm_oceantaco.py
import os
import math
import random
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests, io

import pandas as pd

import matplotlib.pyplot as plt


# pip install diffusers accelerate xarray zarr netcdf4
from diffusers import UNet2DModel, DDPMScheduler

import tacoreader

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

# -----------------------
# 1) EDIT THESE
# -----------------------
tacoreader.use("pandas")
DATA_PATH = "https://huggingface.co/datasets/nilsleh/OceanTACO/resolve/main/"   # e.g. "/path/to/data.zarr" or "/path/to/data.nc"
IS_ZARR = False                              # set False if it's a NetCDF
VAR_NAME = "ssha_filtered"                            # change to your variable name

TRAIN_START_DATE = "2023-05-01"
TRAIN_END_DATE = "2023-05-05"

dataset = tacoreader.load(DATA_PATH)

training_files = dataset.sql(f"""
        SELECT
            "l2:id" AS id,
            REPLACE("l2:internal:gdal_vsi", '/vsicurl/', '') AS url
        FROM l2
        WHERE "l0:stac:time_start" >= '{TRAIN_START_DATE}'
        AND "l0:stac:time_start" <  '{TRAIN_END_DATE}'
        AND "l1:id" LIKE '%NORTH_PACIFIC_EAST%'
        """)

nc_datasets = download_all(training_files)["l3_swot.nc"]

print(f"Loaded {len(nc_datasets)} datasets.")
print("Time range:", nc_datasets[0].attrs["date"], "to", nc_datasets[-1].attrs["date"])

print("Differences between datasets: ")
ds1, ds2 = nc_datasets[0], nc_datasets[1]
ds1, ds2 = xr.align(ds1, ds2, join="exact")  # optional but recommended

num_vars = [v for v in ds1.data_vars
            if (v in ds2.data_vars) and np.issubdtype(ds1[v].dtype, np.number)]

diff = ds1[num_vars] - ds2[num_vars]

plt.figure(figsize=(10, 4))


plt.subplot(1, 3, 1)
diff[VAR_NAME].plot()

plt.subplot(1, 3, 2)
plt.title(f"First time frame of {VAR_NAME}")
nc_datasets[0][VAR_NAME].plot()
# plt.colorbar(fraction=0.046, pad=0.04)

# merge the "track" dimension
ds_merged = []
for ds in nc_datasets:
    if "track" in ds.dims:
        ds_merged.append(ds.max(dim="track"))


plt.subplot(1, 3, 3)
plt.title(f"Track merged frame of {VAR_NAME}")
for i in range(len( ds_merged)):
    ds_merged[i][VAR_NAME].plot()
# plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"dataset_init_{VAR_NAME}.png")

# Crop a small region

LON_MIN, LON_MAX = 130, 150
LAT_MIN, LAT_MAX =  15, 30


# If your variable is something like ds["ssh"].dims == ("time","lat","lon"), this will work.
# If dims differ, adjust indexing in __getitem__.

# -----------------------
# 2) Dataset
# -----------------------
class XrFieldDataset(Dataset):
    def __init__(self, ds: xr.Dataset, var_name: str, size=64, max_items=None):
        self.var = ds[var_name]
        if "time" not in self.var.dims:
            raise ValueError(f"{var_name} must have a 'time' dim. Got dims: {self.var.dims}")

        self.size = size

        # Build valid time indices (drop all-NaN frames)
        t = self.var.sizes["time"]
        valid = []
        for i in range(t):
            arr = self.var.isel(time=i).values
            if np.isfinite(arr).any():
                valid.append(i)

        if max_items is not None:
            valid = valid[:max_items]

        self.idxs = valid

        # quick mean/std estimate from a subset for normalization
        sample_idxs = random.sample(self.idxs, k=min(128, len(self.idxs)))
        vals = []
        for i in sample_idxs:
            a = self.var.isel(time=i).values.astype(np.float32)
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(a)
        vals = np.concatenate(vals) if len(vals) else np.array([0.0], dtype=np.float32)
        self.mean = float(vals.mean())
        self.std = float(vals.std() + 1e-6)

        print(f"[dataset] frames={len(self.idxs)}  mean={self.mean:.4g}  std={self.std:.4g}")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        i = self.idxs[k]
        x = self.var.isel(time=i).values.astype(np.float32)  # (H,W)
        x = np.nan_to_num(x, nan=self.mean)                  # replace NaNs with mean (pre-normalization)
        x = (x - self.mean) / self.std                       # standardize

        # to torch: (1,H,W)
        x = torch.from_numpy(x)[None, ...]

        # resize to fixed size
        x = x[None, ...]  # (1,1,H,W)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = x[0]          # (1,size,size)

        # optional clamp for stability early on
        x = torch.clamp(x, -6.0, 6.0)
        return x


# -----------------------
# 3) Load xarray dataset
# -----------------------
def open_oceantaco(path: str, is_zarr: bool) -> xr.Dataset:
    if is_zarr:
        # common for large gridded data
        return xr.open_zarr(path, consolidated=False)
    else:
        return xr.open_dataset(path)

if VAR_NAME not in nc_datasets[0].data_vars:
    raise KeyError(f"{VAR_NAME} not found. Available vars: {list(nc_datasets[0].data_vars)}")


# merge datasets if there are multiple files (e.g. from different dates)
if len(nc_datasets) > 1:
    print(f"Merging {len(nc_datasets)} datasets...")
    ds_list2 = []
    for ds in nc_datasets:
        date = pd.to_datetime(ds.attrs["date"])   # parse string -> Timestamp
        ds2 = ds.expand_dims(time=[date])         # create a new dimension/coord called "date"
        ds_list2.append(ds2)
    ds = xr.concat(ds_list2, dim="time")
else:
    ds = nc_datasets[0]

ds_cropped = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

print(ds_cropped)

# Plot the dataset


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title(f"First time frame of {VAR_NAME}")
ds[VAR_NAME].isel(time=0).plot()
# plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 2, 2)
plt.title(f"Last time frame of {VAR_NAME}")
ds[VAR_NAME].isel(time=-1).plot()
# plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"dataset_{VAR_NAME}.png")

dataset = XrFieldDataset(ds_cropped, VAR_NAME, size=64, max_items=None)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

# -----------------------
# 4) DDPM model + scheduler
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet2DModel(
    sample_size=64,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

# -----------------------
# 5) Train (small run)
# -----------------------
model.train()
steps = 3000  # enough to see "something" quickly; increase later
log_every = 100
save_every = 1000

it = iter(loader)
for step in range(1, steps + 1):
    try:
        x = next(it)
    except StopIteration:
        it = iter(loader)
        x = next(it)

    x = x.to(device)  # (B,1,64,64)

    # sample random diffusion timesteps
    b = x.shape[0]
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=device).long()

    noise = torch.randn_like(x)
    x_noisy = noise_scheduler.add_noise(x, noise, t)

    # predict noise
    noise_pred = model(x_noisy, t).sample

    loss = F.mse_loss(noise_pred, noise)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % log_every == 0:
        print(f"step {step:5d} | loss {loss.item():.4f}")

    if step % save_every == 0:
        os.makedirs("ckpt", exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "mean": dataset.mean,
                "std": dataset.std,
                "var_name": VAR_NAME,
            },
            f"ckpt/ddpm_{VAR_NAME}_step{step}.pt",
        )

print("done training")

# -----------------------
# 6) Sample
# -----------------------
@torch.no_grad()
def sample(n=8):
    model.eval()
    x = torch.randn(n, 1, 64, 64, device=device)

    for t in noise_scheduler.timesteps:
        # predict noise and step
        noise_pred = model(x, t).sample
        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    # unnormalize back to physical-ish units
    x = x.squeeze(1).cpu().numpy()  # (n,64,64)
    x = x * dataset.std + dataset.mean
    return x

samples = sample(8)
np.save("samples.npy", samples)
print("saved samples.npy")