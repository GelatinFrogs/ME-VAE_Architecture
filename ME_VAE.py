import os
import csv
import umap
from sklearn.manifold import TSNE
from skimage.io import imsave,imread
from skimage.transform import resize
import glob
import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Lambda, Conv2DTranspose, concatenate, Concatenate, Multiply
from keras import optimizers
from keras import metrics
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping

from vae_callback import VAEcallback


os.environ['HDF5_USE_FILE_LOCKING']='FALSE' 


class MEVAE():
    """ 2-dimensional variational autoencoder for latent phenotype capture
    """
    
    def __init__(self, args):
        """ initialize model with argument parameters and build
        """
        self.chIndex        = args.chIndex
        self.data_dir       = args.data_dir
        self.image_dir      = args.image_dir
        self.save_dir       = args.save_dir    
        self.out1_dir        = args.out1_dir
        self.input2_dir     = args.input2_dir
        
        self.use_vaecb      = args.use_vaecb
        self.do_vaecb_each  = args.do_vaecb_each
        self.use_clr        = args.use_clr
        self.earlystop 		= args.earlystop
        
        self.latent_dim     = args.latent_dim
        self.nlayers        = args.nlayers
        self.inter_dim      = args.inter_dim
        self.kernel_size    = args.kernel_size
        self.batch_size     = args.batch_size
        self.epochs         = args.epochs
        self.nfilters       = args.nfilters
        self.learn_rate     = args.learn_rate
        
        self.epsilon_std    = args.epsilon_std
        
        self.latent_samp    = args.latent_samp
        self.num_save       = args.num_save
        
        self.do_tsne        = args.do_tsne
        self.verbose        = args.verbose
        self.phase          = args.phase
        self.steps_per_epoch = args.steps_per_epoch
        
        self.data_size = len(os.listdir(os.path.join(self.data_dir, 'train')))
        self.file_names = os.listdir(os.path.join(self.data_dir, 'train'))
        
        self.image_size     = args.image_size  
        self.nchannel       = args.nchannel
        self.image_res      = args.image_res
        self.show_channels  = args.show_channels
        
        if self.steps_per_epoch == 0:
            self.steps_per_epoch = self.data_size // self.batch_size
                
        self.build_model()
            
        
    def sampling(self, sample_args):
        """ sample latent layer from normal prior
        """
        
        z_mean, z_log_var = sample_args
        
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0],
                                         self.latent_dim),
                                  mean=0,
                                  stddev=self.epsilon_std)
    
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    
    def build_model(self):
        """ build VAE model
        """
        
        input_shape = (self.image_size, self.image_size, self.nchannel)
        
        # build encoder1 model
        inputs1 = Input(shape=input_shape, name='encoder_input1')
        self.inputs1=inputs1
        x1 = inputs1
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            
            x1 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x1)
            filters *= 2
        
        # shape info needed to build decoder model
        shape = K.int_shape(x1)
        
        # generate latent vector Q(z|X)
        x1 = Flatten()(x1)
        x1 = Dense(self.inter_dim, activation='relu')(x1)
        z1_mean = Dense(self.latent_dim, name='z_mean')(x1)
        z1_log_var = Dense(self.latent_dim, name='z_log_var')(x1)
        
        # use reparameterization trick to push the sampling out as input
        z1 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z1')([z1_mean, z1_log_var])
        
        
        # build encoder2 model
        inputs2 = Input(shape=input_shape, name='encoder_input2')
        self.inputs2=inputs2
        x2 = inputs2
        filters = self.nfilters
        kernel_size = self.kernel_size
        for i in range(self.nlayers):
            
            x2 = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x2)
            filters *= 2
            
        x2 = Flatten()(x2)
        x2 = Dense(self.inter_dim, activation='relu')(x2)
        z2_mean = Dense(self.latent_dim, name='z_mean')(x2)
        z2_log_var = Dense(self.latent_dim, name='z_log_var')(x2)
        
        # use reparameterization trick to push the sampling out as input
        z2 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z2')([z2_mean, z2_log_var])
        
        z12=Multiply()([z1,z2])
        
        
        
        # build decoder model
        latent_inputs1 = Input(shape=(self.latent_dim,), name='z_sampling')
        d1 = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs1)
        d1 = Reshape((shape[1], shape[2], shape[3]))(d1)
        
        for i in range(self.nlayers):
            d1 = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                activation='relu',
                                strides=2,
                                padding='same')(d1)
            filters //= 2
        
        
        outputs = Conv2DTranspose(filters=input_shape[2],
                                  kernel_size=kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(d1)
        
        
        
        # instantiate encoder model
        self.encoder1 = Model(inputs1, [z1_mean, z1_log_var, z1], name='encoder1')
        self.encoder1.summary()
        plot_model(self.encoder1, to_file=os.path.join(self.save_dir, 'encoder1_model.png'), show_shapes=True)
        
        # instantiate encoder model
        self.encoder2 = Model(inputs2, [z2_mean, z2_log_var, z2], name='encoder2')
        self.encoder2.summary()
        plot_model(self.encoder2, to_file=os.path.join(self.save_dir, 'encoder2_model.png'), show_shapes=True)
        
        # instantiate decoder1 model
        self.decoder1 = Model(latent_inputs1, outputs, name='decoder1')
        self.decoder1.summary()
        plot_model(self.decoder1, to_file=os.path.join(self.save_dir, 'decoder1_model.png'), show_shapes=True)

        # instantiate VAE model
        outputs1 = self.decoder1(Multiply()([self.encoder1(inputs1)[2], self.encoder2(inputs2)[2]]))
        self.vae = Model(inputs=[inputs1,inputs2], outputs=[outputs1], name='vae')

        
        #   VAE loss terms w/ KL divergence            
        def Decoder1Loss(true, pred):
            xent_loss = metrics.binary_crossentropy(K.flatten(true), K.flatten(pred))
            xent_loss *= self.image_size * self.image_size
            kl_loss = 1 + z1_log_var * 2 - K.square(z1_mean) - K.exp(z1_log_var * 2)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
                        
            vae_loss = K.mean(xent_loss + kl_loss)
            return vae_loss/2
        
        def Decoder2Loss(true, pred):
            xent_loss = metrics.binary_crossentropy(K.flatten(true), K.flatten(pred))
            xent_loss *= self.image_size * self.image_size
            kl_loss = 1 + z2_log_var * 2 - K.square(z2_mean) - K.exp(z2_log_var * 2)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
                        
            vae_loss = K.mean(xent_loss + kl_loss)
            return vae_loss/2
        
        def CombLoss(true,pred):
            l1=Decoder1Loss(true, pred)
            l2=Decoder2Loss(true, pred)
            return l1+l2

        optimizer = optimizers.rmsprop(lr = self.learn_rate)    

        losses = {"decoder1": CombLoss}
        lossWeights = {"decoder1": 1.0}
        
        self.vae.compile(loss=losses,loss_weights=lossWeights, optimizer=optimizer)

        self.vae.summary()       
        plot_model(self.vae, to_file=os.path.join(self.save_dir, 'vae_model.png'), show_shapes=True)

        # save model architectures
        self.model_dir = os.path.join(self.save_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        print('saving model architectures to', self.model_dir)
        with open(os.path.join(self.model_dir, 'arch_vae.json'), 'w') as file:
            file.write(self.vae.to_json())    
        with open(os.path.join(self.model_dir, 'arch_encoder1.json'), 'w') as file:
            file.write(self.encoder1.to_json())
        with open(os.path.join(self.model_dir, 'arch_encoder2.json'), 'w') as file:
            file.write(self.encoder2.to_json())
        with open(os.path.join(self.model_dir, 'arch_decoder1.json'), 'w') as file:
            file.write(self.decoder1.to_json())

    
            
    def train(self):
        """ train VAE model
        """


        #Note: Datagenerator including on the fly rotation, orientation, and size transformations not included here. Implement according to needs and use to load images.
    
        print('Loading Input1 Images')
        imageList=sorted(glob.glob(os.path.join(self.data_dir,'train', '*')))
        
        data = []
        for imagePath in imageList:
                image = imread(imagePath)
                image=resize(image,(self.image_size, self.image_size, self.nchannel))
                image=image*(255/np.max(image))
                data.append(image)
        self.input1_data = np.array(data, dtype="float") / 255.0
            
        print('Loading Input2 Images')
        imageList=sorted(glob.glob(os.path.join(input2_dir,'train', '*')))
        data = []
        for imagePath in imageList:
                image = imread(imagePath)
                image=resize(image,(self.image_size, self.image_size, self.nchannel))
                image=image*(255/np.max(image))
                data.append(image)
        self.input2_data = np.array(data, dtype="float") / 255.0
                                 
                                 
        print('Loading Output Images')
        imageList=sorted(glob.glob(os.path.join(out1_dir,'train', '*')))
        data = []
        for imagePath in imageList:
                image = imread(imagePath)
                image=resize(image,(self.image_size, self.image_size, self.nchannel))
                image=image*(255/np.max(image))
                data.append(image)
        out1_data = np.array(data, dtype="float") / 255.0
            
       
        # instantiate callbacks       
        callbacks = []

        term_nan = TerminateOnNaN()
        callbacks.append(term_nan)

        csv_logger = CSVLogger(os.path.join(self.save_dir, 'training.log'), 
                               separator='\t')
        callbacks.append(csv_logger)
        
        checkpointer = ModelCheckpoint(os.path.join(self.save_dir, 'checkpoints/vae_weights.hdf5'),
                                       verbose=1, 
                                       save_best_only=True,
                                       save_weights_only=True)
        callbacks.append(checkpointer)

        if self.earlystop:
            earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=8)
            callbacks.append(earlystop)

        
        if self.use_vaecb:
            vaecb = VAEcallback(self)
            callbacks.append(vaecb)
        
        
        
        self.history = self.vae.fit(x={"encoder_input1": self.input1_data, "encoder_input2": self.input2_data},y=out1_data,
                                              epochs = self.epochs,
                                              callbacks = callbacks,
                                               batch_size=self.batch_size)


        print('saving model weights to', self.model_dir)
        self.vae.save_weights(os.path.join(self.model_dir, 'weights_vae.hdf5'))
        self.encoder1.save_weights(os.path.join(self.model_dir, 'weights_encoder1.hdf5'))
        self.encoder2.save_weights(os.path.join(self.model_dir, 'weights_encoder2.hdf5'))
        self.decoder1.save_weights(os.path.join(self.model_dir, 'weights_decoder1.hdf5'))

        self.encode()

        print('done!')
   
    

    def encode(self):
        """ encode data with trained model
        """
        
        print('Encoding with Encoder1')
        encoded1= self.encoder1.predict(self.input1_data)
        print('Encoding with Encoder2')
        encoded2= self.encoder2.predict(self.input2_data)
        self.file_names = sorted(list(glob.glob(os.path.join(self.data_dir,'train', '*'))))  
           
         # save generated filename
        fnFile = open(os.path.join(self.save_dir, 'filenames.csv'), 'w')
        with fnFile:
            writer = csv.writer(fnFile)
            for file in self.file_names:
                writer.writerow([file])
        

        encoded_encodings=np.multiply(np.array(encoded1[2]),np.array(encoded2[2]))   
                            
        # generate and save encodings       
        outFile = open(os.path.join(self.save_dir, 'encodings.csv'), 'w')
        with outFile:
            writer = csv.writer(outFile)
            writer.writerows(encoded_encodings)
   
        print('done!')


