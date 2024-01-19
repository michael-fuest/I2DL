"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small


class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvBlock(torch.nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.d = nn.Dropout2d(p=0.1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.d(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class EncoderBlock(torch.nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        # self.d = nn.Dropout2d(p=0.1)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        # p = self.d(p)
        return x, p


class DecoderBlock(torch.nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        # self.d = nn.Dropout2d(p=0.05)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        # x = self.d(x)
        x = self.conv(x)
        return x


class SegmentationNN(torch.nn.Module):

    def __init__(self, hp=None):
        super(SegmentationNN, self).__init__()
        self.hp = hp

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        # Load the pretrained MobileNetV3 model
        self.mobilenetv3_features = mobilenet_v3_small(pretrained=True).features

        # Reducing the number of channels in each layer
        self.conv1 = nn.Conv2d(576, 64, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(32, self.hp["num_classes"], kernel_size=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hp["learning_rate"],
                                          weight_decay=self.hp["weight_decay"])
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = self.mobilenetv3_features(x)

        # Apply additional layers
        x = self.conv1(x)
        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.upsample2(x)
        x = self.conv3(x)

        # If your input image is not the same size as the output,
        # you may need another upsampling step here.
        x = nn.Upsample(size=(240, 240), mode='bilinear', align_corners=True)(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()


if __name__ == "__main__":
    from torchinfo import summary

    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")
