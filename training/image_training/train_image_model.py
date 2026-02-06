import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import sys
import os
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config

# -----------------------------
# Configuration
# -----------------------------
print(f"Training Image Classification Model")
print(f"Dataset: {config.training.IMG_DATASET_PATH}")
print(f"Image Size: {config.model.IMAGE_SIZE}")
print(f"Batch Size: {config.training.IMG_BATCH_SIZE}")
print(f"Epochs: {config.training.IMG_EPOCHS}")
print(f"Classes: {config.model.CLASS_NAMES}")
print("-" * 50)

# -----------------------------
# Data generators
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=config.training.IMG_VALIDATION_SPLIT,
    rotation_range=config.training.ROTATION_RANGE if config.training.USE_AUGMENTATION else 0,
    zoom_range=config.training.ZOOM_RANGE if config.training.USE_AUGMENTATION else 0,
    horizontal_flip=config.training.HORIZONTAL_FLIP if config.training.USE_AUGMENTATION else False,
)

train_generator = datagen.flow_from_directory(
    str(config.training.IMG_DATASET_PATH),
    target_size=config.model.IMAGE_SIZE,
    batch_size=config.training.IMG_BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    str(config.training.IMG_DATASET_PATH),
    target_size=config.model.IMAGE_SIZE,
    batch_size=config.training.IMG_BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = train_generator.num_classes
print(f"Number of classes: {num_classes}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# -----------------------------
# Build Model (Improved CNN Architecture)
# -----------------------------
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation="relu", padding="same", 
           input_shape=(*config.model.IMAGE_SIZE, config.model.IMAGE_CHANNELS)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Dense layers
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        str(config.model.IMAGE_MODEL_PATH),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# -----------------------------
# Train
# -----------------------------
print("\nStarting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=config.training.IMG_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
print("\nEvaluating model...")
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# -----------------------------
# Save model
# -----------------------------
config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save(str(config.model.IMAGE_MODEL_PATH))

print(f"\n✅ Image model saved at: {config.model.IMAGE_MODEL_PATH}")
print("\n" + "="*50)
print("Training completed successfully!")
print("="*50)

