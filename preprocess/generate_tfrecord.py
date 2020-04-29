
from random import shuffle
import numpy as np
import glob
import tensorflow as tf
import cv2
import sys
import os
import json
import data_config
import PIL.Image as Image


def encode_utf8_string(text, length, dic, null_char_id):
    char_ids_padded = [null_char_id]*length
    char_ids_unpadded = [null_char_id]*len(text)
    for i in range(len(text)):
        hash_id = dic[text[i]]
        char_ids_padded[i] = hash_id
        char_ids_unpadded[i] = hash_id
    return char_ids_padded, char_ids_unpadded


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_dataset(path_images, path_label):
	"""
		path_images: path to folder contains all images
		path_label: path to file label json with format {"image_name": "text label"}
		return: list_path_images, label dict
	"""
	all_images = glob.glob(path_images + '/*')
	labels_dict = json.load(open(path_label))

	return all_images, labels_dict


dict={}
with open('chars.txt', encoding="utf") as dict_file:
    for line in dict_file:
        print("--------> Line: ", line)
        char_idx = line.strip().split('\t')
        if len(char_idx) != 2:
            key = char_idx[0]
            print(key)
            dict[" "] = int(key)
        else:
	        key, value = char_idx[0], char_idx[1]
	        dict[value] = int(key)
print((dict))

images_path, labels_dict = get_dataset(data_config.PATH_IMAGE, data_config.PATH_LABEL)

print("------------> Number of images: ", len(images_path))
print("------------> Number of labels", len(labels_dict))

tfrecord_writer  = tf.python_io.TFRecordWriter(data_config.TF_RECORD)
config = tf.ConfigProto()
for j in range(0,int(len(images_path))):
    print('[INFO] Processed images: {}/{}'.format(j,int(len(images_path))))

    sys.stdout.flush()

    img = Image.open(images_path[j])

    img = img.resize(data_config.IMAGE_SIZE, Image.ANTIALIAS)
    np_data = np.array(img)
    image = tf.image.convert_image_dtype(np_data, dtype=tf.uint8)
    image = tf.image.encode_png(image)
    with tf.Session(config=config) as sess:
        image_data = sess.run(image)
        sess.close()

    # get image name
    key = images_path[j].split("/")[-1]
    # get text label of image
    text_label = labels_dict[key]
    #
    char_ids_padded, char_ids_unpadded = encode_utf8_string(
                        text=text_label,
                        dic=dict,
                        length=data_config.MAX_LENGHT,
                        null_char_id=data_config.NUM_NULL_CHAR)

    print("----> char_ids_padded: ", char_ids_padded)
    print("----> char_ids_unpadded: ", char_ids_unpadded)
    print("----> shape: ", np_data.shape)
    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'image/encoded': _bytes_feature(image_data),
                            'image/format': _bytes_feature(b"PNG"),
                            'image/width': _int64_feature([np_data.shape[1]]),
                            'image/orig_width': _int64_feature([np_data.shape[1]]),
                            'image/class': _int64_feature(char_ids_padded),
                            'image/unpadded_class': _int64_feature(char_ids_unpadded),
                            'image/text': _bytes_feature(bytes(text_label, 'utf-8')),
                            # 'height': _int64_feature([crop_data.shape[0]]),
                        }
                    ))
    tfrecord_writer.write(example.SerializeToString())
tfrecord_writer.close()

sys.stdout.flush()