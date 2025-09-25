from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    directory="artifacts/data_ingestion/lung_colon_ct_scan_image_set/Train_and_Validation_Set",
    subset="training",
    target_size=(224, 224),
    batch_size=32,
    shuffle=True
)

print("Class indices:", train_generator.class_indices)