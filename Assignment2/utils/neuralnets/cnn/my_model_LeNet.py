import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras import Model

class LeNet(Model):
    """
    LeNet is an early and famous CNN architecture for image classfication task.
    It is proposed by Yann LeCun. Here we use its architecture as the startpoint
    for your CNN practice. Its architecture is as follow.

    input >> Conv2DLayer >> Conv2DLayer >> flatten >>
    DenseLayer >> AffineLayer >> softmax loss >> output

    Or

    input >> [conv2d-avgpooling] >> [conv2d-avgpooling] >> flatten >>
    DenseLayer >> AffineLayer >> softmax loss >> output

    http://deeplearning.net/tutorial/lenet.html
    """

    def __init__(self, input_shape, output_size=10):
        '''
        input_shape: The size of the input. (img_len, img_len, channel_num).
        output_size: The size of the output. It should be equal to the number of classes.
        '''
        super(LeNet, self).__init__()
        #############################################################
        # TODO: Define layers for your custom LeNet network         
        # Hint: Try adding additional convolution and avgpool layers
        #############################################################
        self.conv_layer_1 = Conv2D(filters=20, kernel_size=(5, 5), strides=(1,1), activation='relu', input_shape=input_shape, padding="same")
        self.avgpool_layer_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.conv_layer_2 = Conv2D(filters=50, kernel_size=(5, 5), strides=(1,1), activation='relu', input_shape=input_shape, padding="same")
        self.avgpool_layer_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.flatten_layer = Flatten()
        self.fc_layer_1 = Dense(500, activation='relu')

        self.fc_layer_2 = Dense(output_size, activation='softmax')

        #############################################################
        #                          END TODO                         #                                              
        #############################################################
        
        
    def call(self, x):
        '''
        x: input to LeNet model.
        '''
        #call function returns forward pass output of the network
        #############################################################
        # TODO: Implement forward pass for custom network defined 
        # in __init__ and return network output
        #############################################################
        x = self.conv_layer_1(x)
        x = self.avgpool_layer_1(x)

        x = self.conv_layer_2(x)
        x = self.avgpool_layer_2(x)

        x = self.flatten_layer(x)
        x = self.fc_layer_1(x)

        out = self.fc_layer_2(x)
        
        return out
        #############################################################
        #                          END TODO                         #                                              
        #############################################################
        
        

