"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
from torch.optim import Adam

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architectures, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.hparams["input_channels"], 32, kernel_size=self.hparams["kernel_size"],
                      padding=self.hparams["padding"]),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(self.hparams["pool_size"], self.hparams["pool_stride"]),

            nn.Conv2d(32, 64, kernel_size=self.hparams["kernel_size"], padding=self.hparams["padding"]),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(self.hparams["pool_size"], self.hparams["pool_stride"]),

        )

        # Adaptive pooling layer to flatten the output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        # Fully-connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(self.hparams["dropout"]),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.hparams["dropout"]),

            nn.Linear(512, self.hparams["output_neurons"])  # Output layer
        )

        self.optimizer = torch.optim.Adam(self.parameters())
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
