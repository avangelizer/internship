from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
import pandas as pd
import numpy as np

class FineTuneCNN:
    """Generic class that uses Xception, InceptionV3, VGG19, VGG16, or ResNet50 model pre-trained on imagenet weights 
    to train, test, and predict image classifications.
    
    Attributes:
        model_name (str) : name of the model architecture
        dimensions (tup) : target dimensions to resize images to
        model_arch (keras.models.Model) : Xception, InceptionV3, VGG19, VGG16, or ResNet50 model
        num_classes (int) : number of classes for classification
        lr (float) : learning rate for the optimizer
        train_after_layer (int) : the layer after which you want to start training the model
        train_path (str) : path to the training data directory; must be nested by class
        num_training_samples (int) : number of training data samples
        validation_path (str) : path to the validation data directory; must be nested by class
        num_validation_samples (int) : number of validation data samples
        batch_size (int) : number of data samples to train on per batch
        num_epochs (int) : number epochs to train
        aug (bool) : True to augment the data, default is False
        crop (bool) : True to crop the data, default is False
        trn_gen (keras.preprocessing.image.ImageDataGenerator) : train data image generator
        val_gen (keras.preprocessing.image.ImageDataGenerator) : validation data image generator
        test_gen (keras.preprocessing.image.ImageDataGenerator) : test data image generator
        early_stopping (bool) : True to stop training early based on the validation loss, default is True 
        early_stop (keras.callbacks.Callback) : EarlyStopping callback from Keras
        model (keras.models.Model) : the instantiated model to be trained, tested, etc.
        test_path (str) : path to the testing data directory; must be nested by class
        num_test_samples (int) : number of testing data samples
        inference_path (str) : path to the inference data directory; must be nested within one folder
        num_inference_samples (int) : number of inference data samples
        checkpoint (keras.callbacks.Callback) : ModelCheckpoint callback from Keras
        model_path (str) : saved model path
        n_crops: number of crops taken from image
    """
    
    def __init__(self, model_name=None, num_classes=None, lr=None):
        """Defines the target image size for the model based on the model architecture.
        """
        self.model_name = model_name
        
        assert model_name in ["VGG16","VGG19","ResNet50","Xception","InceptionV3"]
        if self.model_name in ["VGG16","VGG19","ResNet50"]:
            self.dimensions=(224,224)
        else:
            self.dimensions=(299,299)
        
    def __preprocess(self, x):
        """Preprocesses the image data based on the model architecture
        
        Returns:
            numpy.array : Normalized image data
        """
        if self.model_name in ["VGG16","VGG19","ResNet50"]:
            x[:, :, 0] -= 103.939
            x[:, :, 1] -= 116.779
            x[:, :, 2] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, :, ::-1]
            return x
        else:
            x /= 255.
            x -= 0.5
            x *= 2.
            return x
        
    def build(self, model_arch=None, lr=None, num_classes=None, train_after_layer=-1):
        """Build the model
        
        Args:
            model_arch (keras.models.Model) : Xception, InceptionV3, VGG19, VGG16, or ResNet50 model
            lr (float) : learning rate for the optimizer
            num_classes (int) : number of classes for classification 
            train_after_layer (int) : the layer after which you want to start training the model
        """
        self.model_arch = model_arch
        self.lr = lr
        self.num_classes = num_classes
        self.train_after_layer = train_after_layer
        
        # create the base pre-trained model
        base_model = self.model_arch
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # and a logistic layer
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # this is the model we will train
        self.model = Model(input=base_model.input, output=predictions)
        
        for layer in self.model.layers[:self.train_after_layer]:
           layer.trainable = False
        for layer in self.model.layers[self.train_after_layer:]:
           layer.trainable = True
        
        self.model.compile(optimizer=SGD(lr=self.lr, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
        print('Model built.')
    ##########ADDED TO CROP IMAGES########
    def random_crop(img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]
    def crop_generator(self,xy_gen):
        """
        Take as input a Keras ImageGen (Iterator) and generate random
        crops from the image batches generated by the original iterator.
        """
        while True:
            x, y = next(xy_gen)
            
            for i in range(x.shape[0]):
                #for each image repeat n_crops times
                batch_crops = np.zeros((self.n_crops, self.dimensions[0],  self.dimensions[0], 3))
                batch_ys = np.tile(y, (self.n_crops, 1))  # repeat 'y' 5 times
                for j in range(self.n_crops):
                    batch_crops[j] = random_crop(x[i], self.dimensions)
                yield (batch_crops, batch_ys) #yield the n_crops from the same image

    
               
    def train(self, train_path=None, num_training_samples=None, validation_path=None, num_validation_samples=None, 
              batch_size=0, num_epochs=0, early_stopping=False, aug=False,crop=False,n_crops=1 ,checkpoint=None, model_path=None):
        """Fits the model
        
        Args:
            train_path (str) : path to the training data directory; must be nested by class
            num_training_samples (int) : number of training data samples
            validation_path (str) : path to the validation data directory; must be nested by class
            num_validation_samples (int) : number of validation data samples
            batch_size (int) : number of data samples to train on per batch
            num_epochs (int) : number epochs to train
            early_stopping (bool) : True to stop training early based on the validation loss, default is True
            aug (bool) : True to augment the data, default is False
            checkpoint (keras.callbacks.Callback) : ModelCheckpoint callback from Keras
        """
        self.train_path = train_path # training data needs to be within folders per class label
        self.num_training_samples = num_training_samples
        self.validation_path = validation_path
        self.num_validation_samples = num_validation_samples
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        self.aug = aug
        self.crop = crop
        self.n_crops=n_crops
        if self.aug:
            if self.crop:
                train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                               height_shift_range=0.05, shear_range=0.05,
                                               channel_shift_range=.1,horizontal_flip=True, 
                                               preprocessing_function=self.__preprocess)
            else:
                train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                               height_shift_range=0.05, shear_range=0.05,
                                               channel_shift_range=.1,horizontal_flip=True, target_size=self.dimensions,
                                               preprocessing_function=self.__preprocess)
                
        else:
            if not self.crop:
                train_datagen = ImageDataGenerator(target_size=self.dimensions, preprocessing_function=self.__preprocess)
            else:
                train_datagen = ImageDataGenerator(preprocessing_function=self.__preprocess)

        self.trn_gen = train_datagen.flow_from_directory(self.train_path,
                                                         batch_size=self.batch_size, class_mode='categorical', shuffle=True, seed=42)
        if self.crop:
                self.trn_gen = self.crop_generator(self.trn_gen)
        if validation_path:
            val_datagen = ImageDataGenerator(preprocessing_function=self.__preprocess)

            self.val_gen = val_datagen.flow_from_directory(self.validation_path,  
                                                             batch_size=self.batch_size, target_size=self.dimensions,
                                                             class_mode='categorical', shuffle=True, seed=42)
            
        else:
            self.val_gen = None
        
        self.early_stopping = early_stopping
        if self.early_stopping:
            self.early_stop = EarlyStopping(monitor='val_loss', min_delta=0.10, patience=0, verbose=0, mode='auto')
            callback_list = [self.early_stop]
        else:
            callback_list = []
        
        self.checkpoint = checkpoint
        if self.checkpoint:
            callback_list.append(self.checkpoint)
            
        self.model.fit_generator(self.trn_gen, samples_per_epoch=self.num_training_samples * self.n_crops, 
                                 validation_data=self.val_gen, nb_val_samples=self.num_validation_samples,
                                 nb_epoch=self.num_epochs, verbose=1, callbacks=callback_list,workers=1,use_multiprocessing=False)
    
    def test(self, test_path=None, num_test_samples=None, batch_size=32):
        """Evaluates the model's performance on loss and accuracy
        
        Args:
            test_path (str) : path to the testing data directory; must be nested by class
            num_test_samples (int) : number of testing data samples
            batch_size (int) : number of data samples per batch
        Returns:
            float : loss
            float : accuracy
            numpy.array : ground truth class labels
        
        """
        self.test_path = test_path # images need to be nested in a folder in this directory
        self.num_test_samples = num_test_samples
        self.batch_size = batch_size
        test_datagen = ImageDataGenerator(preprocessing_function=self.__preprocess)
        self.test_gen = test_datagen.flow_from_directory(self.test_path, 
                                                         batch_size=self.batch_size, target_size=self.dimensions,class_mode="categorical", 
                                                         shuffle=False, seed=42)


        loss,accuracy = self.model.evaluate_generator(self.test_gen, val_samples=self.num_test_samples)
        
        return loss, accuracy, self.test_gen.classes
    
    def load_weights(self, model_path=None):
        """Loads weights for a pre-trained model.
        
        Args:
            model_path (str) : saved model path
        """
        self.model = load_model(model_path)
    
    def inference(self, inference_path=None, num_inference_samples=None, batch_size=0, class_labels=None):
        """Predicts the probability of an image belonging to a certain class
        
        Args:
            inference_path (str) : path to the inference data directory; must be nested within one folder
            num_inference_samples (int) : number of inference data samples
            batch_size (int) : number of data samples per batch
            class_labels (list) : ordered list of class labels
        Returns:
            pandas.DataFrame : a dataframe of the images and their softmax probabilities
        """
        self.inference_path = inference_path # images need to be nested in a folder in this directory
        self.num_inference_samples = num_inference_samples
        self.batch_size = batch_size
        
        pred_datagen = ImageDataGenerator(preprocessing_function=self.__preprocess)
        pred_gen = pred_datagen.flow_from_directory(self.inference_path, target_size=self.dimensions, batch_size=self.batch_size,
                                                    class_mode=None, shuffle=False, seed=42)

        predictions, filenames = self.model.predict_generator(pred_gen, val_samples=self.num_inference_samples), pred_gen.filenames

        prediction_list = []
        for i in range(len(filenames)):
            image = [filenames[i].split("/")[-1]]
            preds = predictions[i,:].tolist()
            prediction_list.append(image + preds)
        
        try:
            class_indices = self.trn_gen.class_indices
        except:
            class_indices = {value: index for index, value in enumerate(class_labels)}
        class_labels_ordered = [""]*len(class_indices)
        for label,index in class_indices.items():
            class_labels_ordered[index] = label
        class_labels_ordered = ["image"] + class_labels_ordered
        
        inference_df = pd.DataFrame(prediction_list,columns=class_labels_ordered)
        
        return inference_df
