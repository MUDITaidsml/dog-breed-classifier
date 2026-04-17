import os
import logging
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 60
EPOCHS = 40
DATA_DIR = '../data'
MODEL_DIR = '../model'

def build_model(num_classes):
    logger.info("Building ResNet50V2 Transfer Learning Model...")
    # Load ResNet50V2 pre-trained on ImageNet, excluding top fully connected layers
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

def get_data_generators():
    logger.info("Initializing ImageDataGenerator with aggressive augmentation...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Note: Expects directories 'data/train' and 'data/val' to exist with subfolders for each breed
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    return train_generator, val_generator

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # If data doesn't exist, exit gracefully instructing the user
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        logger.error(f"Data directory '{DATA_DIR}/train' not found. Please download the dataset and organize it.")
        return

    train_generator, val_generator = get_data_generators()
    model, base_model = build_model(num_classes=train_generator.num_classes)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=os.path.join(MODEL_DIR, 'resnet50v2_dog_breed.h5'), monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # Phase 1: Train top layers only
    logger.info("PHASE 1: Training the custom classification head...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        callbacks=callbacks
    )

    # Phase 2: Fine-tuning (Unfreeze top 30 layers of ResNet50V2)
    logger.info("PHASE 2: Unfreezing top 30 layers of ResNet50V2 for Fine-Tuning...")
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # Recompile with a very low learning rate for fine-tuning
    model.compile(
        optimizer=RMSprop(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("Resuming training for fine-tuning...")
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS - 15,
        callbacks=callbacks
    )

    logger.info("Training complete! Model saved to model/resnet50v2_dog_breed.h5")

if __name__ == '__main__':
    main()
