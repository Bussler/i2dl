"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn.functional as F


class SegmentationNN(pl.LightningModule):
    def __init__(self, train_set, val_set, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.train_set = train_set
        self.val_set = val_set

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64, 
                kernel_size=(3, 3),
                stride=1
            ),
            hparams["activation"],
            nn.Conv2d(
                in_channels=64,
                out_channels=64, 
                kernel_size=(3, 3),
                stride=1
            ),
            hparams["activation"],
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2)
            ),
            nn.Dropout(
                p=hparams["p_dropout"]
            ),
            # 240 -> 118
            nn.Conv2d(
                in_channels=64,
                out_channels=128, 
                kernel_size=(3, 3),
                stride=1
            ),
            hparams["activation"],
            nn.Conv2d(
                in_channels=128,
                out_channels=128, 
                kernel_size=(3, 3),
                stride=1
            ),
            hparams["activation"],
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2)
            ),
            nn.Dropout(
                p=hparams["p_dropout"]
            ),
            # 118 -> 57
            nn.Conv2d(
                in_channels=128,
                out_channels=256, 
                kernel_size=(3, 3),
                stride=1
            ),
            hparams["activation"],
            nn.Conv2d(
                in_channels=256,
                out_channels=256, 
                kernel_size=(2, 2),
                stride=1
            ),
            hparams["activation"],
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2)
            ),
            nn.Dropout(
                p=hparams["p_dropout"]
            ),
            # 57 -> 27
            nn.Conv2d(
                in_channels=256,
                out_channels=512, 
                kernel_size=(1, 1),
                stride=1
            ),
            hparams["activation"],
            nn.Conv2d(
                in_channels=512,
                out_channels=1024, 
                kernel_size=(1, 1),
                stride=1
            ),
            hparams["activation"]
            # 27 -> 27
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=(1, 1),
                stride=1
            ),
            hparams["activation"],
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=(1, 1),
                stride=1
            ),
            hparams["activation"],
            # 27 -> 27
            nn.Upsample(
                scale_factor=2
            ),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(2, 2),
                stride=1
            ),
            hparams["activation"],
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1
            ),
            hparams["activation"],
            nn.Dropout(
                p=hparams["p_dropout"]
            ),
            # 27 -> 57
            nn.Upsample(
                scale_factor=2
            ),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1
            ),
            hparams["activation"],
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1
            ),
            hparams["activation"],
            nn.Dropout(
                p=hparams["p_dropout"]
            ),
            # 57 -> 118
            nn.Upsample(
                scale_factor=2
            ),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1
            ),
            hparams["activation"],
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1
            ),
            hparams["activation"],
            nn.Dropout(
                p=hparams["p_dropout"]
            ),
            # 118 -> 240
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(1, 1),
                stride=1
            ),
            hparams["activation"],
            nn.Conv2d(
                in_channels=32,
                out_channels=num_classes,
                kernel_size=(1, 1),
                stride=1
            ),
            hparams["activation"]
        )

    def general_step(self, batch, batch_idx, mode):  
        images, targets = batch
        # forward pass
        outputs = self.forward(images)
        # loss
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(outputs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        optim = None
        if(self.hparams["optimizer"] == "SGD"):
            optim = optim.SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.hparams["learning_rate"], momentum=self.hparams["momentum"])
        elif(self.hparams["optimizer"] == "Adam"):
            optim = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.hparams["learning_rate"])
        else:
            optim = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.hparams["learning_rate"])
        return optim

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

        if(self.is_cuda):
            x = x.cuda()
        else:
            x = x.cpu()
        x = self.encoder(x)
        x = self.decoder(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".
    
        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class DummySegmentationModel(pl.LightningModule):

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