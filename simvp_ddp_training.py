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

#pre-computed global normalisation stats
mean_ssh = 0.074
std_ssh = 0.0986
mean_sst = 293.307
std_sst = 8.726

@tf.function
def load_and_preprocess_batch(file_path):
    try:
        dataset = tf.data.TFRecordDataset(file_path)
        dataset = dataset.map(parse_example)
    except:
        tf.print('File: '+file_path+' is corrupted')
        dataset = tf.data.Dataset.from_tensor_slices([]) 
    
    return dataset

@tf.function
def parse_example(serialized_example):
    feature_description = {
        'input': tf.io.FixedLenFeature(int(batch_size*(n_t+1)*n*n*2), tf.float32),
        'output': tf.io.FixedLenFeature(int(batch_size*n_t*n_obs_max*3), tf.float32)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    input_data = tf.reshape(example['input'], [batch_size,n_t,n,n,2])
    input_data = normalise_ssh(input_data)
    input_data = tf.transpose(tf.reshape(input_data,[batch_size,n_t,n,n,1]),perm=[0,1,4,2,3])

    output_data = tf.reshape(example['output'], [batch_size,n_t,n_obs_max,3])

    x = output_data[:,:,:,0]
    x_new = rescale_x(x)
    y = output_data[:,:,:,1]
    y_new = rescale_y(y)
    sla = output_data[:,:,:,2]
    sla_new = normalise_ssh(sla)

    outvar = tf.stack((x_new,y_new,sla_new),axis = -1)

    return input_data, outvar

@tf.function
def parse_example_sst(serialized_example):
    feature_description = {
        'input': tf.io.FixedLenFeature(int(batch_size*(n_t+1)*n*n*2), tf.float32),
        'output': tf.io.FixedLenFeature(int(batch_size*n_t*n_obs_max*3), tf.float32)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    input_data = tf.reshape(example['input'], [batch_size,n_t+1,n,n,2])
    input_data = input_data[:, :n_t, :, :, :]  # drop the unused extra timestep

    input_data_ssh = normalise_ssh(input_data[:,:,:,:,0])
    input_data_sst = normalise_sst(input_data[:,:,:,:,1])
    input_data = tf.transpose(tf.stack((input_data_ssh,input_data_sst),axis=-1),perm=[0,1,4,2,3])
    output_data = tf.reshape(example['output'], [batch_size,n_t,n_obs_max,3])

    x = output_data[:,:,:,0]
    x_new = rescale_x(x)
    y = output_data[:,:,:,1]
    y_new = rescale_y(y)
    sla = output_data[:,:,:,2]
    sla_new = normalise_ssh(sla)

    outvar = tf.stack((x_new,y_new,sla_new),axis = -1)

    return input_data, outvar

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


class SSH_Dataset(Dataset):
    def __init__(self, tfrecord_paths):
        self.tfrecord_paths = tfrecord_paths

    def __len__(self):
        return len(self.tfrecord_paths)

    def __getitem__(self, idx):

        serialized_example = tf.data.TFRecordDataset(self.tfrecord_paths[idx])
        parsed_example = serialized_example.map(parse_example_sst)  # Parse the example
        invar, outvar = next(iter(parsed_example))  # Extract the tensors from the parsed example
        invar = torch.from_numpy(invar.numpy())
        outvar = torch.from_numpy(outvar.numpy())
        
        
        return invar, outvar

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
    
train_dir = './pre-processed/training/'
val_dir = './pre-processed/validation/'

weight_dir = './model_weights/'
log_dir = './loss_logs/'
viz_dir = './model_preds/'

n_t = 30
L_x = 960e3
L_y = 960e3
n = 128
batch_size = 25 # DON'T CHANGE, THIS IS FIXED IN THE PRE-PROCESSING TO BE 1 BATCH PER FILE
n_obs_max = 400 # max number of SSH observations on any day in loss function, allows to have fixed size inputs/outputs with zero padding making it easier to create TFRecord dataset
n_train_samples = 1000000
experiment_name = f'simvp_ssh_sst_ns{n_train_samples}_global_'
num_epochs = 50
workers_per_gpu = 1 # sets the number of CPU processes used per GPU to parallelise the data loading/pre-processing
            
frames = n_t

def check_batch(invar, outvar, batch_idx, verbose=True):
    """
    Returns True if batch is clean, False if it has issues.
    """
    issues = []

    # Check for NaN/Inf in inputs
    if torch.isnan(invar).any():
        issues.append(f"NaN in input: {torch.isnan(invar).sum().item()} values out of {invar.numel()}")
    if torch.isinf(invar).any():
        issues.append(f"Inf in input: {torch.isinf(invar).sum().item()} values out of {invar.numel()}")

    # Check for NaN/Inf in outputs
    if torch.isnan(outvar).any():
        issues.append(f"NaN in output: {torch.isnan(outvar).sum().item()} values out of {outvar.numel()}")
    if torch.isinf(outvar).any():
        issues.append(f"Inf in output: {torch.isinf(outvar).sum().item()} values out of {outvar.numel()}")

    # Check input value ranges (after normalisation, expect roughly -5 to 5)
    nonzero_input = invar[invar != 0]
    if len(nonzero_input) > 0:
        if verbose:
            print(f"  Input  | nonzero: {len(nonzero_input):>10} | "
                  f"min={nonzero_input.min():.3f} max={nonzero_input.max():.3f} "
                  f"mean={nonzero_input.mean():.3f} std={nonzero_input.std():.3f}")
    else:
        issues.append("Input is ALL ZEROS")

    # Check output SSH values (column 2, should be normalised ~-5 to 5)
    sla = outvar[:, :, :, 2]
    nonzero_sla = sla[sla != 0]
    if len(nonzero_sla) > 0:
        if verbose:
            print(f"  Output SLA | nonzero: {len(nonzero_sla):>8} | "
                  f"min={nonzero_sla.min():.3f} max={nonzero_sla.max():.3f} "
                  f"mean={nonzero_sla.mean():.3f}")
    else:
        issues.append("Output SLA is ALL ZEROS (no observations?)")

    # Check x/y coordinate ranges (should be 0 to 127 after rescaling)
    x_coords = outvar[:, :, :, 0]
    y_coords = outvar[:, :, :, 1]
    nonzero_x = x_coords[x_coords != 0]
    nonzero_y = y_coords[y_coords != 0]
    if len(nonzero_x) > 0:
        if verbose:
            print(f"  Output x   | nonzero: {len(nonzero_x):>8} | "
                  f"min={nonzero_x.min():.1f} max={nonzero_x.max():.1f}")
            print(f"  Output y   | nonzero: {len(nonzero_y):>8} | "
                  f"min={nonzero_y.min():.1f} max={nonzero_y.max():.1f}")
    else:
        issues.append("Output x/y coords are ALL ZEROS (no observations?)")

    # Fraction of zero-padded observations
    obs_mask = (outvar[:, :, :, 2] != 0)
    frac_valid = obs_mask.float().mean().item()
    if frac_valid < 0.01:
        issues.append(f"Only {frac_valid*100:.2f}% of observations are non-zero — very sparse")
    elif verbose:
        print(f"  Valid obs fraction: {frac_valid*100:.1f}%")

    if issues:
        print(f"[Batch {batch_idx}] ISSUES FOUND:")
        for issue in issues:
            print(f"  !! {issue}")
        return False
    elif verbose:
        print(f"[Batch {batch_idx}] OK")
    return True


def run_pre_training_checks(train_loader, val_loader, rank, n_batches=5):
    """
    Run before training. Checks first n_batches from train and val loaders.
    """
    print("\n" + "="*60)
    print("PRE-TRAINING DATA CHECKS")
    print("="*60)

    all_clean = True

    print(f"\n-- Checking {n_batches} training batches --")
    for i, (invar, outvar) in enumerate(train_loader):
        if i >= n_batches:
            break
        invar = invar.squeeze(0).to(rank)
        outvar = outvar.squeeze(0).to(rank)
        print(f"\nBatch {i} | input shape: {tuple(invar.shape)} | output shape: {tuple(outvar.shape)}")
        clean = check_batch(invar, outvar, i)
        if not clean:
            all_clean = False

    print(f"\n-- Checking {n_batches} validation batches --")
    for i, (invar, outvar) in enumerate(val_loader):
        if i >= n_batches:
            break
        invar = invar.squeeze(0).to(rank)
        outvar = outvar.squeeze(0).to(rank)
        print(f"\nBatch {i} | input shape: {tuple(invar.shape)} | output shape: {tuple(outvar.shape)}")
        clean = check_batch(invar, outvar, i)
        if not clean:
            all_clean = False

    print("\n" + "="*60)
    if all_clean:
        print("All checks passed. Safe to train.")
    else:
        print("WARNING: Issues found. Investigate before training.")
    print("="*60 + "\n")

    return all_clean

def train(rank, world_size, checkpoint_path=None):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    lr = 0.001
    n_train_batches = int(n_train_samples/batch_size)
    n_val_batches = 500

    #SSH-SST:
    model = SimVP_Model_no_skip_sst(in_shape=(n_t,2,128,128),model_type='gsta',hid_S=8,hid_T=128,drop=0.2,drop_path=0.15).to(rank)
    
    #SSH ONLY:
    # model = SimVP_Model_no_skip(in_shape=(n_t,1,128,128),model_type='gsta',hid_S=8,hid_T=128,drop=0.2,drop_path=0.15).to(rank)

    train_files = os.listdir(train_dir)
    train_dataset_files = [train_dir+f for f in train_files if '.tfrecord' in f]
    train_dataset_files = train_dataset_files[:n_train_batches]
    n_train_batches=len(train_dataset_files)
    train_dataset = SSH_Dataset(train_dataset_files)

    val_files = os.listdir(val_dir)
    val_dataset_files = [val_dir+f for f in val_files if '.tfrecord' in f]
    val_dataset_files = val_dataset_files[:n_val_batches]
    val_dataset = SSH_Dataset(val_dataset_files)
    
    viz_files = os.listdir(val_dir)
    viz_dataset_files = [val_dir+f for f in viz_files if '.tfrecord' in f]
    viz_dataset_files = viz_dataset_files[:4]
    viz_dataset = SSH_Dataset(viz_dataset_files)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    viz_sampler = torch.utils.data.distributed.DistributedSampler(viz_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=workers_per_gpu, sampler=train_sampler)

    print("Train loader: ", train_loader)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=workers_per_gpu, sampler=val_sampler)
    viz_loader = torch.utils.data.DataLoader(viz_dataset, num_workers=workers_per_gpu, sampler=viz_sampler)
    
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

    # --- ADD THIS after building loaders ---
    if rank == 0:
        clean = run_pre_training_checks(train_loader, val_loader, rank, n_batches=5)
        if not clean:
            print("Aborting training due to data issues.")
            dist.destroy_process_group()
            return

    for epoch in range(start_epoch, num_epochs):
        #training loop
        model.train()
        train_loss = 0.0
        num_batches=0
        
        for torch_input_batch, torch_output_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            torch_input_batch = torch_input_batch.squeeze(0).to(rank)
            torch_output_batch = torch_output_batch.squeeze(0).to(rank)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(torch_input_batch)
                loss = criterion(outputs, torch_output_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Epoch {epoch} Batch {num_batches}] NaN/Inf loss detected — skipping batch")
                # Check if it's the model outputs or the targets causing it
                print(f"  model output: nan={torch.isnan(outputs).sum().item()} inf={torch.isinf(outputs).sum().item()}")
                print(f"  target:       nan={torch.isnan(torch_output_batch).sum().item()} inf={torch.isinf(torch_output_batch).sum().item()}")
                print(f"  scaler scale: {scaler.get_scale()}")
                continue  # skip this batch rather than propagating NaN through weights

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # guard against exploding gradients
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            num_batches += 1

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
    num_processes = 1
    print(f'Number of GPUs used: {num_processes}')

    mp.spawn(train, args=(num_processes,), nprocs=num_processes,)  # add checkpoint file name here if restarting training