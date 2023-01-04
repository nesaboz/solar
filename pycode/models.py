import torch.nn as nn
import torch
import numpy as np


def get_padding(kernel_size, stride, width=256):
    """
    Padding that ensures the output_size of ceil(input_size/stride). Assuming square images:
    """
    output_width = np.ceil(width / stride)
    padding = int(np.ceil(((output_width-1) * stride - width + kernel_size) / 2))
    return padding


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, x):
        return self.model(x)


class Down(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )


class DownNoActivation(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )


class Up(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channels)
        )


class DoubleDown(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.model = nn.Sequential(
            Down(in_channels, out_channels, kernel_size, stride, padding),
            DownNoActivation(out_channels, out_channels, kernel_size, stride, padding)
        )


class ResBranch(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.model = DownNoActivation(in_channels, out_channels, kernel_size, stride, padding)


class ResBlock(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.double_down = DoubleDown(in_channels, out_channels, kernel_size, stride, padding)
        self.res_branch = ResBranch(in_channels, out_channels, 1, stride, 0)
        self.elu = nn.ELU()

    def forward(self, x):
        trunk = self.double_down(x)
        branch = self.res_branch(x)
        x = torch.cat([trunk, branch], 1)  # this step doubles the number of channels
        x = self.elu(x)
        return x


class Segnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down_1 = Down(n_channels, 32, 5, 2, 2)
        self.res_block_1 = ResBlock(32, 32, 3, 1, 1)

        self.down_2 = Down(64, 64, 5, 2, 2)
        self.res_block_2 = ResBlock(64, 64, 3, 1, 1)

        self.down_3 = Down(128, 128, 5, 2, 2)
        self.res_block_3 = ResBlock(128, 128, 3, 1, 1)

        self.up_1 = Up(256, 64, 5, 2, 2, 1)
        self.elu_1 = nn.ELU()

        self.up_2 = Up(192, 32, 5, 2, 2, 1)
        self.elu_2 = nn.ELU()

        self.final_conv = nn.ConvTranspose2d(96, self.n_classes, 5, 2, 2, 1)

    def forward(self, x):
        x = self.down_1(x)
        div2 = self.res_block_1(x)

        x = self.down_2(div2)
        div4 = self.res_block_2(x)

        x = self.down_3(div4)
        div8 = self.res_block_3(x)

        x = self.up_1(div8)
        added = torch.cat([x, div4], 1)
        x = self.elu_1(added)

        x = self.up_2(x)
        added = torch.cat([x, div2], 1)
        x = self.elu_2(added)

        x = self.final_conv(x)
        return x


# def test_segnet():
    # segnet = Segnet(n_channels=3, n_classes=2)
    # x = torch.randn([32, 3, 256, 256])
    # yhat = segnet(x)


# import tensorflow as tf
#
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import BatchNormalization, Activation, Lambda, Add, Conv2DTranspose
# from tensorflow.keras.layers import Input, Conv2D
# from tensorflow.keras.models import Model
#
#
# def make_segnet_residual_block(block_input, depth, kernel_size, stride):
#     x = Conv2D(depth, kernel_size, stride, padding="SAME")(block_input)
#     x = BatchNormalization()(x)
#     x = Activation("elu")(x)
#     x = Conv2D(depth, kernel_size, stride, padding="SAME")(x)
#     x = BatchNormalization()(x)
#     branch = Conv2D(depth, 1, stride, padding="SAME")(block_input)
#     branch = BatchNormalization()(branch)
#     added = Add()([x, branch])
#     x = Activation("elu")(added)
#     return x
#
# #
# def segnet(input_shape, n_classes=3, deploy=False):
#     if deploy:
#         input_shape = (None, None, 3)
#         input_placeholder = Input(tensor=K.placeholder(dtype=tf.float32, shape=input_shape, name="input_place"))
#         inputs = Lambda(lambda x: K.expand_dims(x, axis=0))(input_placeholder)
#     else:
#         inputs = Input(dtype=tf.float32, shape=input_shape, name="input_place")
#
#     # block1
#     x = Conv2D(32, 5, 2, padding="SAME")(inputs)
#     x = BatchNormalization()(x)
#     x = Activation("elu")(x)
#
#     # res_block1
#     div2 = make_segnet_residual_block(x, 32, 3, 1)
#
#     # block2
#     x = Conv2D(64, 5, 2, padding="SAME")(div2)
#     x = BatchNormalization()(x)
#     x = Activation("elu")(x)
#
#     # res_block2
#     div4 = make_segnet_residual_block(x, 64, 3, 1)
#
#     # block3
#     x = Conv2D(128, 5, 2, padding="SAME")(div4)
#     x = BatchNormalization()(x)
#     x = Activation("elu")(x)
#     div8 = make_segnet_residual_block(x, 128, 3, 1)
#
#     x = Conv2DTranspose(64, 5, 2, padding="SAME")(div8)
#     x = BatchNormalization()(x)
#     added = Add()([x, div4])
#     x = Activation("elu")(added)
#
#     x = Conv2DTranspose(32, 5, 2, padding="SAME")(x)
#     x = BatchNormalization()(x)
#     added = Add()([x, div2])
#     x = Activation("elu")(added)
#
#     x = Conv2DTranspose(n_classes, 5, 2, padding="SAME")(x)
#     predictions = Activation("softmax", name="output")(x)
#     single_predictions = Lambda(lambda x: K.squeeze(x, 0), name="single_inference_output")(predictions)
#
#     if deploy:
#         model = Model(inputs=input_placeholder, outputs=single_predictions)
#     else:
#         model = Model(inputs=inputs, outputs=predictions)
#     return model
