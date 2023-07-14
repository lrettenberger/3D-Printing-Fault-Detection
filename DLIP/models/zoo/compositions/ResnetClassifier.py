from typing import List, Tuple
import torch
import torch.nn as nn
import wandb
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from DLIP.data.base_classes.segmentation.base_seg_dataset import BaseSegmentationDataset

from DLIP.models.zoo.compositions.base_composition import BaseComposition
from DLIP.models.zoo.decoder.unet_decoder import UnetDecoder
from DLIP.models.zoo.encoder.example_encoder import ExampleEncoder
from DLIP.models.zoo.encoder.resnet_encoder import ResNetEncoder
from DLIP.models.zoo.encoder.unet_encoder import UnetEncoder
from torch.utils.data import DataLoader


from DLIP.utils.data_preparation.custom_collate import custom_collate   
from DLIP.utils.data_preparation.seed_worker import seed_worker


class ResnetClassifier(BaseComposition):
    
    def __init__(
        self,
        loss_fcn: nn.Module
    ):
        super().__init__()
        self.append(ResNetEncoder(
            input_channels = 3,
            encoder_type = 'resnet18',
            pretraining_weights = 'imagenet',
            classification_output = True,
        ))
        self.loss_fcn = nn.CrossEntropyLoss()
        self.val_confusion_matrix = None
        self.test_confusion_matrix = None
        
        silver_dataset = BaseSegmentationDataset(
                    root_dir='/home/ws/kg2371/projects/3D-Printing-Fault-Detection/data/silver_test', 
                    transforms=None,
        )
        self.silver_test_loader = DataLoader(
            silver_dataset,
            batch_size=256,
            collate_fn=custom_collate,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=seed_worker,
            shuffle=True,
            drop_last=False
        )
        
        oblique_dataset = BaseSegmentationDataset(
                    root_dir='/home/ws/kg2371/projects/3D-Printing-Fault-Detection/data/oblique_test', 
                    transforms=None,
        )
        self.oblique_test_loader = DataLoader(
            oblique_dataset,
            batch_size=256,
            collate_fn=custom_collate,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=seed_worker,
            shuffle=True,
            drop_last=False
        )
 
    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", torch.sum(torch.argmax(y_pred,dim=1) == y_true) / y_true.numel(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", torch.sum(torch.argmax(y_pred,dim=1) == y_true) / y_true.numel(), prog_bar=True)
        
        
        # if batch_idx == 0:
        #     self.val_confusion_matrix = self.calc_confusion(torch.argmax(y_pred,dim=1),y_true)
        # elif batch_idx != 6:
        #     self.val_confusion_matrix += self.calc_confusion(torch.argmax(y_pred,dim=1),y_true)
        # else:
        #     # normalize
        #     sums = torch.zeros_like(self.val_confusion_matrix)
        #     sums[0,:] = self.val_confusion_matrix[0].sum()
        #     sums[1,:] = self.val_confusion_matrix[1].sum()
        #     sums[2,:] = self.val_confusion_matrix[2].sum()
        #     self.val_confusion_matrix = (self.val_confusion_matrix / sums).cpu()
        #     labels = ['Good','Under-Extrusion','Stringing']
        #     wandb.log({'confusion_matrix': wandb.plots.HeatMap(labels, labels, self.val_confusion_matrix, show_text=True)})
        
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss_n_c = self.loss_fcn(y_pred, y_true)
        loss = torch.mean(loss_n_c)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test/acc", torch.sum(torch.argmax(y_pred,dim=1) == y_true) / y_true.numel(), prog_bar=True)
        
        if batch_idx == 0:
            self.test_confusion_matrix = self.calc_confusion(torch.argmax(y_pred,dim=1),y_true)
        elif batch_idx != 6:
            self.test_confusion_matrix += self.calc_confusion(torch.argmax(y_pred,dim=1),y_true)
        else:
            # normalize
            sums = torch.zeros_like(self.test_confusion_matrix)
            sums[0,:] = self.test_confusion_matrix[0].sum()
            sums[1,:] = self.test_confusion_matrix[1].sum()
            sums[2,:] = self.test_confusion_matrix[2].sum()
            self.test_confusion_matrix = (self.test_confusion_matrix / sums).cpu()
            labels = ['Good','Under-Extrusion','Stringing']
            wandb.log({'confusion_matrix_test': wandb.plots.HeatMap(labels, labels, self.test_confusion_matrix, show_text=True)})
        
            # silverbed
            y_true = torch.zeros(0).cuda()
            y_pred = torch.zeros(0).cuda()
            for batch in self.silver_test_loader:
                x,y_true_step = batch 
                y_pred_step = self.forward(x.cuda())
                y_true = torch.concat((y_true.cuda(),y_true_step.cuda()))
                y_pred = torch.concat((y_pred.cuda(),torch.argmax(y_pred_step,dim=1).cuda()))
            confmat = ConfusionMatrix(num_classes=3,normalize='true').cuda()
            confmat_silberbed = confmat(y_pred.to(torch.int),y_true.to(torch.int))
            wandb.log({'confusion_matrix_silverbed': wandb.plots.HeatMap(labels, labels, confmat_silberbed.cpu(), show_text=True)})

            # oblique
            y_true = torch.zeros(0).cuda()
            y_pred = torch.zeros(0).cuda()
            for batch in self.oblique_test_loader:
                x,y_true_step = batch 
                y_pred_step = self.forward(x.cuda())
                y_true = torch.concat((y_true.cuda(),y_true_step.cuda()))
                y_pred = torch.concat((y_pred.cuda(),torch.argmax(y_pred_step,dim=1).cuda()))
            confmat = ConfusionMatrix(num_classes=3,normalize='true').cuda()
            confmat_oblique = confmat(y_pred.to(torch.int),y_true.to(torch.int))
            wandb.log({'confusion_matrix_oblique': wandb.plots.HeatMap(labels, labels, confmat_oblique.cpu(), show_text=True)})

        
        
        return loss
    
    def calc_confusion(self,preds,targets):
        confmat = ConfusionMatrix(num_classes=3).cuda()
        return confmat(preds,targets)
        
        # labels = ['Good','Under-Extrusion','Stringing']
        # wandb.log({'confusion_matrix': wandb.plots.HeatMap(labels, labels, matrix.cpu(), show_text=True)})
