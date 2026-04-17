import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)

def load_and_preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Read using OpenCV
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to ResNet50 input size
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    
    # Scale pixels to [0, 1] as done in ImageDataGenerator
    img_scaled = img_resized / 255.0
    
    # Expand dimensions to match batch size (1, 224, 224, 3)
    img_batch = np.expand_dims(img_scaled, axis=0)
    return img_rgb, img_batch

def predict_breed(model_path, image_path, class_names=None):
    logger.info("Loading trained ResNet50V2 model...")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train the model first.")
        return

    model = tf.keras.models.load_model(model_path)
    
    logger.info(f"Processing image: {image_path}")
    original_img, processed_img = load_and_preprocess_image(image_path)
    
    # Prediction
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    label = f"Class ID: {predicted_class_idx}"
    if class_names:
        label = class_names[predicted_class_idx]
        
    logger.info(f"\n--- Prediction Results ---")
    logger.info(f"Predicted Breed: {label}")
    logger.info(f"Confidence Score: {confidence * 100:.2f}%\n")
    
    # Display the result
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.title(f"Prediction: {label} ({confidence*100:.1f}%)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Dog Breed using trained CNN")
    parser.add_argument('--image', type=str, required=True, help="Path to the dog image")
    parser.add_argument('--model', type=str, default='../model/resnet50v2_dog_breed.h5', help="Path to the saved model")
    args = parser.parse_args()
    
    predict_breed(args.model, args.image)
