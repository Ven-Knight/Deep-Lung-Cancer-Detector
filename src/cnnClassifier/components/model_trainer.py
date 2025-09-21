# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Standard Libraries
# ────────────────────────────────────────────────────────────────────────────────────────
import os
import urllib.request as request
import tensorflow     as tf
import time
from   zipfile    import ZipFile
from   pathlib    import Path

# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Project Modules for config entity
# ────────────────────────────────────────────────────────────────────────────────────────
from cnnClassifier.entity.config_entity import TrainingConfig        # Typed config object

# ────────────────────────────────────────────────────────────────────────────────────────
# Training Class: Handles model loading, data generators, and training execution
# ────────────────────────────────────────────────────────────────────────────────────────
class Training:
    def __init__(self, config: TrainingConfig):
        """
        Initialize with structured config containing paths and hyperparameters.

        Args:
            config (TrainingConfig): Configuration entity for training stage.
        """
        self.config = config

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Load Updated Base Model
    # ────────────────────────────────────────────────────────────────────────────────────────
    def get_base_model(self):
        """
        Loads the updated base model (with custom layers) from disk.
        """
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Setup Training and Validation Data Generators
    # ────────────────────────────────────────────────────────────────────────────────────────
    def train_valid_generator(self):
        """
        Creates training and validation data generators using ImageDataGenerator.
        Applies augmentation if enabled in config.
        """
        # Common preprocessing parameters
        datagenerator_kwargs = dict(
                                        rescale          = 1./255,
                                        validation_split = 0.20
                                   )

        # Image resizing and batching parameters
        dataflow_kwargs      = dict(
                                        target_size   = self.config.params_image_size[:-1],  # Exclude channel dimension
                                        batch_size    = self.config.params_batch_size,
                                        interpolation = "bilinear"
                                   )

        # Validation generator (no augmentation)
        valid_datagenerator  = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
                                                                        directory = self.config.training_data,
                                                                        subset    = "validation",
                                                                        shuffle   = False,
                                                                        **dataflow_kwargs
                                                                      )

        # Training generator (with optional augmentation)
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                                                                                    rotation_range     = 40,
                                                                                    horizontal_flip    = True,
                                                                                    width_shift_range  = 0.2,
                                                                                    height_shift_range = 0.2,
                                                                                    shear_range        = 0.2,
                                                                                    zoom_range         = 0.2,
                                                                                    **datagenerator_kwargs
                                                                                 )
        else:
            train_datagenerator = valid_datagenerator                 # Use same generator without augmentation

        self.train_generator    = train_datagenerator.flow_from_directory(
                                                                                directory = self.config.training_data,
                                                                                subset    = "training",
                                                                                shuffle   = True,
                                                                                **dataflow_kwargs
                                                                         )

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Save Trained Model to Disk
    # ────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the trained model to the specified path.

        Args:
            path (Path): Destination path for saving the model.
            model (tf.keras.Model): Trained model instance.
        """
        model.save(path)

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Train Model Using Generators
    # ────────────────────────────────────────────────────────────────────────────────────────
    def train(self):
        """Trains the model in two phases: head training and fine-tuning."""
        self.steps_per_epoch  = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Phase 1: Train classification head
        self.model.compile(
                            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate_head),
                            loss      = 'categorical_crossentropy',
                            metrics   = ['accuracy']
                          )
        print("Training with frozen base model...")
        self.model.fit(
                            self.train_generator,
                            epochs           = self.config.params_epochs_head,
                            steps_per_epoch  = self.steps_per_epoch,
                            validation_steps = self.validation_steps,
                            validation_data  = self.valid_generator
                      )

        # Phase 2: Fine-tune top layers
        
        # Locate the embedded base model by type
        # Extract base model by slicing known layers
        base_model = tf.keras.models.Model(
                                            inputs  = self.model.input,
                                            outputs = self.model.get_layer("block5_pool").output  # Last layer of VGG16
                                          )
        # Apply fine-tuning logic
        base_model.trainable = True
        for layer in base_model.layers[:-self.config.params_freeze_till]:
            layer.trainable = False


        self.model.compile(
                            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate_fine),
                            loss      = 'categorical_crossentropy',
                            metrics   = ['accuracy']
                          )

        early_stop = tf.keras.callbacks.EarlyStopping    (patience=5, restore_best_weights=True)
        reduce_lr  = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)

        print("Fine-tuning top layers...")
        self.model.fit(
                            self.train_generator,
                            epochs           = self.config.params_epochs_fine,
                            steps_per_epoch  = self.steps_per_epoch,
                            validation_steps = self.validation_steps,
                            validation_data  = self.valid_generator,
                            callbacks        = [early_stop, reduce_lr]
                        )

        # Save to artifacts/training (which will be ignored by gitignore)
        self.save_model(path=self.config.trained_model_path, model=self.model)

        # Save to model/final_model.h5 (tracked outside .gitignore)
        export_path = Path(self.config.model_export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure model/ exists
        self.save_model(path=export_path, model=self.model)

