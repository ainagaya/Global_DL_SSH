import tensorflow as tf

path = "./pre-processed/training/batch_0.tfrecord"

raw_ds = tf.data.TFRecordDataset(path)
raw_record = next(iter(raw_ds))

example = tf.train.Example()
example.ParseFromString(raw_record.numpy())

print(example)