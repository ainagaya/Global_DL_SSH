import tensorflow as tf
import torch
from torch.utils.data import Dataset

# @tf.function
# def parse_example_sst(serialized_example):
#     feature_description = {
#         'input': tf.io.FixedLenFeature(int(batch_size*n_t*n*n*2), tf.float32),
#         'output': tf.io.FixedLenFeature(int(batch_size*n_t*n_obs_max*3), tf.float32)
#     }
#     try:
#         example = tf.io.parse_single_example(serialized_example, feature_description)

#         input_data = tf.reshape(example['input'], [batch_size,n_t,n,n,2])
#         input_data_ssh = normalise_ssh(input_data[:,:,:,:,0])
#         input_data_sst = normalise_sst(input_data[:,:,:,:,1])
#         input_data = tf.transpose(tf.stack((input_data_ssh,input_data_sst),axis=-1),perm=[0,1,4,2,3])
#         output_data = tf.reshape(example['output'], [batch_size,n_t,n_obs_max,3])

#         x = output_data[:,:,:,0]
#         x_new = rescale_x(x)
#         y = output_data[:,:,:,1]
#         y_new = rescale_y(y)
#         sla = output_data[:,:,:,2]
#         sla_new = normalise_ssh(sla)

#         outvar = tf.stack((x_new,y_new,sla_new),axis = -1)
#     except:
#         tf.print('File is corrupted')
#         input_data = tf.zeros([batch_size,n_t,2,n,n],tf.float32)
#         outvar = tf.zeros([batch_size,n_t,n_obs_max,3],tf.float32)
        

#     return input_data, outvar



#pre-computed global normalisation stats
mean_ssh = 0.074
std_ssh = 0.0986
mean_sst = 293.307
std_sst = 8.726

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
def parse_example_sst(serialized_example):
    feature_description = {
        'input': tf.io.FixedLenFeature(int(batch_size*(n_t+1)*n*n*2), tf.float32),
        'output': tf.io.FixedLenFeature(int(batch_size*n_t*n_obs_max*3), tf.float32)
    }
    try:
        example = tf.io.parse_single_example(serialized_example, feature_description)

        input_data = tf.reshape(example['input'], [batch_size,n_t,n,n,2])
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
    except Exception as e:
        tf.print('File is corrupted')
        print(e)
        input_data = tf.zeros([batch_size,n_t,2,n,n],tf.float32)
        outvar = tf.zeros([batch_size,n_t,n_obs_max,3],tf.float32)
        

    return input_data, outvar

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
        parsed_example = serialized_example.map(parse_example_sst)
        invar, outvar = next(iter(parsed_example))
        invar = torch.from_numpy(invar.numpy())
        outvar = torch.from_numpy(outvar.numpy())
        return invar, outvar


# ---- put your files here ----
tfrecord_paths = [
    "./pre-processed/training/batch_0.tfrecord",
    "./pre-processed/training/batch_1.tfrecord",
    "./pre-processed/training/batch_2.tfrecord",
    "./pre-processed/training/batch_3.tfrecord",
]

dataset = SSH_Dataset(tfrecord_paths)

print(f"Checking {len(dataset)} TFRecord files...\n")

for i in range(len(dataset)):
    path = tfrecord_paths[i]
    try:
        x, y = dataset[i]
        print(f"[OK]   {path}")
        print(f"       input : shape={tuple(x.shape)}, dtype={x.dtype}")
        print(f"       output: shape={tuple(y.shape)}, dtype={y.dtype}")
    except StopIteration:
        print(f"[FAIL] {path}")
        print("       file opened but contains no records")
    except Exception as e:
        print(f"[FAIL] {path}")
        print(f"       {type(e).__name__}: {e}")