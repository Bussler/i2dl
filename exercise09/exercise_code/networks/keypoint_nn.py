"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super().__init__()
        #super(KeypointModel, self).__init__()
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
        ########################################################################

        self.conv_kernel = lambda n_in, n_out: n_in - n_out + 1
        self.pool_kernel = lambda n_in, n_out: int(n_in / n_out)
        ck = self.conv_kernel
        pk = self.pool_kernel
        self.dropout_prob = 0.1

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=hparams["input_channels"], 
                out_channels=hparams["dim_conv2d_1"][0], 
                kernel_size=(ck(hparams["input_size"], hparams["dim_conv2d_1"][1]), ck(hparams["input_size"], hparams["dim_conv2d_1"][2])),
                stride=1
            ),

            hparams["activation"],

            nn.MaxPool2d(
                kernel_size=(pk(hparams["dim_activation_1"][1], hparams["dim_maxpooling2d_1"][1]), pk(hparams["dim_activation_1"][2], hparams["dim_maxpooling2d_1"][2])),
                stride=(pk(hparams["dim_activation_1"][1], hparams["dim_maxpooling2d_1"][1]), pk(hparams["dim_activation_1"][2], hparams["dim_maxpooling2d_1"][2]))
            ),

            nn.Dropout(
                p=self.dropout_prob
            ),

            nn.Conv2d(
                in_channels=hparams["dim_dropout_1"][0], 
                out_channels=hparams["dim_conv2d_2"][0], 
                kernel_size=(ck(hparams["dim_dropout_1"][1], hparams["dim_conv2d_2"][1]), ck(hparams["dim_dropout_1"][2], hparams["dim_conv2d_2"][2])),
                stride=1
            ),

            hparams["activation"],

            nn.MaxPool2d(
                kernel_size=(pk(hparams["dim_activation_2"][1], hparams["dim_maxpooling2d_2"][1]), pk(hparams["dim_activation_2"][2], hparams["dim_maxpooling2d_2"][2])),
                stride=(pk(hparams["dim_activation_2"][1], hparams["dim_maxpooling2d_2"][1]), pk(hparams["dim_activation_2"][2], hparams["dim_maxpooling2d_2"][2]))
            ),

            nn.Dropout(
                p=self.dropout_prob + hparams["p_dropout_incr"]
            ),

            nn.Conv2d(
                in_channels=hparams["dim_dropout_2"][0], 
                out_channels=hparams["dim_conv2d_3"][0], 
                kernel_size=(ck(hparams["dim_dropout_2"][1], hparams["dim_conv2d_3"][1]), ck(hparams["dim_dropout_2"][2], hparams["dim_conv2d_3"][2])),
                stride=1
            ),

            hparams["activation"],

            nn.MaxPool2d(
                kernel_size=(pk(hparams["dim_activation_3"][1], hparams["dim_maxpooling2d_3"][1]), pk(hparams["dim_activation_3"][2], hparams["dim_maxpooling2d_3"][2])),
                stride=(pk(hparams["dim_activation_3"][1], hparams["dim_maxpooling2d_3"][1]), pk(hparams["dim_activation_3"][2], hparams["dim_maxpooling2d_3"][2]))
            ),

            nn.Dropout(
                p=self.dropout_prob + 2 * hparams["p_dropout_incr"]
            ),

            nn.Conv2d(
                in_channels=hparams["dim_dropout_3"][0], 
                out_channels=hparams["dim_conv2d_4"][0], 
                kernel_size=(ck(hparams["dim_dropout_3"][1], hparams["dim_conv2d_4"][1]), ck(hparams["dim_dropout_3"][2], hparams["dim_conv2d_4"][2])),
                stride=1
            ),

            hparams["activation"],

            nn.MaxPool2d(
                kernel_size=(pk(hparams["dim_activation_4"][1], hparams["dim_maxpooling2d_4"][1]), pk(hparams["dim_activation_4"][2], hparams["dim_maxpooling2d_4"][2])),
                stride=(pk(hparams["dim_activation_4"][1], hparams["dim_maxpooling2d_4"][1]), pk(hparams["dim_activation_4"][2], hparams["dim_maxpooling2d_4"][2]))
            ),

            nn.Dropout(
                p=self.dropout_prob + 3 * hparams["p_dropout_incr"]
            ),

            nn.Flatten(),

            nn.Linear(
                hparams["dim_flatten_1"],
                hparams["dim_dense_1"]
            ),

            hparams["activation"],

            nn.Dropout(
                p=self.dropout_prob + 4 * hparams["p_dropout_incr"]
            ),

            nn.Linear(
                hparams["dim_dropout_5"],
                hparams["dim_dense_2"]
            ),

            nn.Linear(
                hparams["dim_dense_2"],
                hparams["dim_linear_1"]
            ),

            nn.Dropout(
                p=self.dropout_prob + 5 * hparams["p_dropout_incr"]
            ),

            nn.Linear(
                hparams["dim_dropout_6"],
                hparams["output_size"]
            )
        )
        
    def training_step(self, batch, batch_idx):
        image, keypoints = batch["image"], batch["keypoints"]
        output = self.forward(image)
        target = keypoints.view(self.hparams["batch_size"], 15, 2)
        prediction = output.view(self.hparams["batch_size"], 15, 2)
        loss = 0
        for i in range(self.hparams["batch_size"]):
            loss += self.hparams["loss"](target[i], prediction[i])
        return {"loss": loss}

    def configure_optimizers(self):
        if(self.hparams["optimizer"] == "Adam"):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        return optimizer

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
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
