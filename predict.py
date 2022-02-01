
import argparse # this will get all functionalities to build your command line command 
import time 
import numpy as np
import matplotlib.pyplot as plt
import sys
### for tensorflow and image pre-processing
from PIL import Image
###
import json ## for the classes file label.map 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub




batch_size = 64
image_size = 224
class_names = {}
# Note: I used the same code in the notebook for moist parts 
# TODO: Create the process_image function
# this function do the same image preprocessing and normalizing for all images 
def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32) # or use image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_n):
    # open the image in the given path 
    openImage = Image.open(image_path)
    # convert te image to an array and do the image preprocessing defined before (1- convert array to 
    # tensor 2- unite the size for all images 3- normalize)
    image = np.asarray(openImage)
    processed_image = process_image(image)
    # add extra dimention for batch size
    processed_image_extra_dimention = np.expand_dims(processed_image, axis=0)
   
    predicted_image = model.predict(processed_image_extra_dimention)
    probabilities = - np.partition(-predicted_image[0], top_n)[:top_n]
    classes = np.argpartition(-predicted_image[0], top_n)[:top_n]
    return probabilities, list(classes)


if __name__ == '__main__':
    print('Enter Your Arguments')
    
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument ('--image_dir', default='./test_images/hard-leaved_pocket_orchid.jpg', help = 'Path to image.', type = str)
    parser.add_argument('--model', help='Trained Model.', type=str)
    parser.add_argument ('--top_k', default = 5, help = 'Top K most likely classes.', type = int)
    parser.add_argument ('--classes' , default = 'label_map.json', help = 'Mapping of categories to real names.', type = str)

    args = parser.parse_args()
    print(args)
    print('arg1:', args.image_dir)
    print('arg2:', args.model)
    print('arg3:', args.top_k)
    print('arg3:', args.classes)

    #1
    # hard-leaved_pocket_orchid.jpg
    image_path = args.image_dir
    #2
    # ./saved.h5
    model = tf.keras.models.load_model(args.model ,custom_objects={'KerasLayer':hub.KerasLayer} ,compile=False)
    #model = tf.keras.models.load_model(save_dir, custom_objects = {'KerasLayer': hub.KerasLayer},compile=False)
    #3
    # 
    top_k = args.top_k
    if top_k is None: 
        top_k = 5
    #4
    # 'label_map.json'
    with open(args.classes, 'r') as f:
        class_names = json.load(f)
    
    class_labels = []
    probs, classes = predict(image_path, model, top_k)
    for i in classes:
        class_labels.append(class_names[str(i)])
    
   
    
    print(probs)
    print(classes)
    print(class_labels)
    # this is to get the index of the maximum probability 
    index_max = max(range(len(probs)), key=probs.__getitem__)
    #print(index_max)
    print("the image is most probably {} with a probality of {}".format(class_labels[index_max] , probs[index_max]))
    
    # python predict.py --image_dir './test_images/cautleya_spicata.jpg'  --model './saved.h5' --top_k 5 --classes './label_map.json'