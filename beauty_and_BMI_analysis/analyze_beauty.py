import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import joblib
import caffe  # Ensure caffe is imported

# Import functions from forward.py
from forward import load_img, get_mean_npy

def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haarcascade_path = './analyze_beauty_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    cropped_face = image[y:y + h, x:x + w]
    return cropped_face, (x, y, w, h)

def preprocess_image(image, means, batch_shape):
    image = load_img(image, resize=(256, 256), isColor=True, crop_size=batch_shape[3], crop_type='center_crop', raw_scale=255, means=means)
    return image

def analyze_beauty(face, net, batch_shape):
    inputs = preprocess_image(face, means, batch_shape)
    net.blobs['data'].data[...] = inputs
    output = net.forward().values()[0][0][0]
    return output

def main():
    # Define paths
    model_deploy_path = './CNN_beauty_predict/trained_models_for_caffe/resnext50_deploy.prototxt'
    model_weights_path = './CNN_beauty_predict/trained_models_for_caffe/models/resnext50.caffemodel'
    mean_file_path = './CNN_beauty_predict/data/1/256_train_mean.binaryproto'
    image_path = "./temp_face.jpg"

    # Load the mean file
    means = get_mean_npy(mean_file_path, crop_size=(224, 224), isColor=True)

    # Load the Caffe model
    caffe.set_mode_gpu()
    net = caffe.Net(model_deploy_path, model_weights_path, caffe.TEST)

    # Load and process the image
    image = cv2.imread(image_path)
    face, _ = detect_and_crop_face(image)
    if face is not None:
        beauty_score = analyze_beauty(face, net, batch_shape=(1, 3, 224, 224))
        print(f"Predicted beauty score: {beauty_score}")
    else:
        print("No face detected")

if __name__ == "__main__":
    main()
