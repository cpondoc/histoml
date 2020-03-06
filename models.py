from tensorflow.keras.applications import *
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

""" Creates a model from scratch """
def build_model(conv_base):
    
    # Built with fully connected layer classifier
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    conv_base.trainable = False
    return model

""" Returns each pre-trained model """
def return_pretrained(name):
    
    # Width and Height of Model
    img_width, img_height = 425, 256
    
    # Case of VGG-16
    if (name == "VGG-16"):
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    # Case of Xception
    elif (name == "Xception"):
        conv_base = Xception(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
        
    # Case of VGG-19
    elif (name == "VGG-19"):
        conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    # Case of ResNetV2
    elif (name == "ResNetV2"):
        conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    # Case of InceptionV3
    elif (name == "InceptionV3"):
        conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    # Case of InceptionResNetV2
    elif (name == "InceptionResNetV2"):
        conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    # Case of Mobile Net
    elif (name == "MobileNet"):
        conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    # Case of MobileNetV2
    elif (name == "MobileNetV2"):
        conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    # Case of DenseNet
    elif (name == "DenseNet"):
        conv_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    # Case of NAsNetLarge
    elif (name == "NASNetLarge"):
        conv_base = NASNetLarge(weights='imagenet', include_top=False, input_shape=(331, 331, 3))
        
    return conv_base

""" Creates the generators to be returned """
def create_generators(validation_dir, train_dir, test_dir, img_width, img_height):
    
    # Create the train data generator
    train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
    
    # Create the test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Train generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')
    
    # Validation generator
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')
    
    # Test Generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')
    
    return train_generator, validation_generator, test_generator