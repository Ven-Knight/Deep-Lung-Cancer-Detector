# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Standard Libraries
# ────────────────────────────────────────────────────────────────────────────────────────
import os
import urllib.request as request
import tensorflow     as tf
from   zipfile        import ZipFile
from   pathlib        import Path

# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Project Modules
# ────────────────────────────────────────────────────────────────────────────────────────
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig  # Typed config object

# ────────────────────────────────────────────────────────────────────────────────────────
# PrepareBaseModel Class: Loads and customizes pretrained CNN architecture
# ────────────────────────────────────────────────────────────────────────────────────────
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initialize with structured config containing model paths and hyperparameters.

        Args:
            config (PrepareBaseModelConfig): Configuration entity for base model setup.
        """
        self.config = config

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Load Pretrained Base Model (VGG16)
    # ────────────────────────────────────────────────────────────────────────────────────────
    def get_base_model(self):
        """
        Loads the base model dynamically based on config.base_model_type.
        Saves the raw base model to disk for reproducibility.
        """        
        if self.config.base_model_type.lower() == "vgg16":
            base_model = tf.keras.applications.VGG16(
                                                        input_shape = self.config.params_image_size,
                                                        weights     = self.config.params_weights,
                                                        include_top = self.config.params_include_top                                                        
                                                    )
        else:
            raise ValueError(f"Unsupported base model type: {self.config.base_model_type}")

        self.base_model = base_model                    # Store separately for later use

        self.save_model(path=self.config.base_model_path, model=base_model)


    # ────────────────────────────────────────────────────────────────────────────────────────
    # Prepare Full Model: Adds custom layers and compiles the model
    # ────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Freezes layers as per config, adds classification head, and compiles the model.

        Args:
            model (tf.keras.Model)    : Pretrained base model.
            classes (int)             : Number of output classes.
            freeze_all (bool)         : Whether to freeze all layers.
            freeze_till (int or None) : Number of layers to keep trainable from the end.
            learning_rate (float)     : Learning rate for optimizer.

        Returns:
            tf.keras.Model            : Fully prepared and compiled model.
        """
 
        model.trainable = True
        if freeze_all:
            model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Add custom classification head including droupout
        x           = tf.keras.layers.Flatten()                                                                          (model.output)
        x           = tf.keras.layers.Dense  (256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x           = tf.keras.layers.Dropout(0.5)                                                                       (x)
        prediction  = tf.keras.layers.Dense  (classes, activation='softmax')                                             (x)


        full_model  = tf.keras.models.Model(
                                            inputs  = model.input,
                                            outputs = prediction
                                           )

        # Compile with SGD optimizer and categorical crossentropy loss
        full_model.compile(
                            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss      = tf.keras.losses.CategoricalCrossentropy(),
                            metrics   = ["accuracy"]
                          )

        full_model.summary()
        return full_model

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Update Base Model: Apply custom layers and save updated model
    # ────────────────────────────────────────────────────────────────────────────────────────
    def update_base_model(self):
        """
        Prepares the full model by adding classification layers and freezing base layers.
        Saves the updated model to disk.
        """        
        self.full_model = self._prepare_full_model(
                                                    model         = self.base_model,
                                                    classes       = self.config.params_classes,
                                                    freeze_all    = self.config.params_freeze_all,
                                                    freeze_till   = self.config.params_freeze_till,
                                                    learning_rate = self.config.params_learning_rate
                                                  )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Save Model to Disk
    # ────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the given Keras model to the specified path.

        Args:
            path (Path): Destination path for saving the model.
            model (tf.keras.Model): Model to be saved.
        """
        model.save(path)