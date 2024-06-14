import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
import torch
import pandas as pd
import os
from losses.softtarget import SoftTargetCrossEntropy


class BaseModule(pl.LightningModule):
    def __init__(self, model, hparams=None, **kwargs):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # self.save_hyperparameters()
        self.model = model(**vars(hparams)) if hparams is not None else model(**kwargs)

        hparams = self.model.hparams
        self.num_classes = hparams.num_classes
        self.dataset_name = hparams.dataset_artifact

        # Create model
        self.lr = hparams.lr
        self.weight_decay = hparams.weight_decay
        self.lr_head = hparams.lr_head
        self.weight_decay_head = hparams.weight_decay_head
        self.t_max_scheduler = (
            hparams.t_max_scheduler if hasattr(hparams, "t_max_scheduler") else 100
        )

        # Create loss module
        if hparams.label_smoothing and hparams.mixup:
            # handled by mixup
            self.train_loss = SoftTargetCrossEntropy()
        else:
            self.train_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        # Create metrics
        # drive act dataset uses accuracy macro
        self.val_acc_micro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="micro"
        )
        self.val_acc_macro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            num_classes=hparams.num_classes, average="macro", task="multiclass"
        )
        self.best_val_acc_micro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="micro"
        )
        self.best_val_acc_macro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="macro"
        )
        self.best_val_f1 = torchmetrics.F1Score(
            num_classes=hparams.num_classes, average="macro", task="multiclass"
        )
        self.test_acc_micro = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="micro"
        )
        self.test_acc_topk = torchmetrics.Accuracy(
            task="multiclass", num_classes=hparams.num_classes, average="micro", top_k=5
        )
        self.test_f1 = torchmetrics.F1Score(
            num_classes=hparams.num_classes, average="macro", task="multiclass"
        )
        self.validation_step_outputs = {"preds": [], "labels": []}

    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.model.hparams.freeze_backbone:
            optimizer = optim.AdamW(
                [
                    {
                        "params": self.model.head.parameters(),
                        "lr": self.lr_head,
                        "weight_decay": self.weight_decay_head,
                    },
                ],
            )
        else:
            if getattr(self.model.hparams, "use_dual_learning_rate", 1):
                params = [
                    {
                        "params": self.model.net.parameters(),
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": self.model.head.parameters(),
                        "lr": self.lr_head,
                        "weight_decay": self.weight_decay_head,
                    },
                ]
                # if lr_patch_embed exists add to params
                if getattr(self, "lr_patch_embed", None):
                    params.append(
                        {
                            "params": self.model.patch_embed.parameters(),
                            "lr": self.lr_patch_embed,
                            "weight_decay": self.weight_decay_patch_embed,
                        },
                    )
            else:
                params = [
                    {
                        "params": self.model.parameters(),
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    },
                ]

            optimizer = optim.AdamW(params)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max_scheduler
        )
        # CosineAnnealingWarmRestarts
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        if isinstance(batch, torch._utils.ExceptionWrapper):
            print("batch is ExceptionWrapper", batch.exc_msg)
            batch.reraise()

        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.train_loss(preds, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.val_loss(preds, labels)
        # calculate metrics
        self.val_acc_micro(preds, labels)
        self.val_acc_macro(preds, labels)
        self.val_f1(preds, labels)

        # save preds and labels for later
        self.validation_step_outputs["preds"].append(preds)
        self.validation_step_outputs["labels"].append(labels)
        # Add sync_dist=True to sync logging across all GPU workers (may have performance impact)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_acc_micro",
            self.val_acc_micro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_acc_macro",
            self.val_acc_macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        # log best val loss
        best_val_loss = self.trainer.callback_metrics.get("best_val_loss")
        outputs = self.all_gather(self.validation_step_outputs)
        # merge the outputs
        preds = None
        for data in outputs["preds"]:
            if preds is None:
                preds = torch.Tensor(data).view(-1, self.num_classes)
            else:
                preds = torch.cat(
                    (preds, torch.Tensor(data).view(-1, self.num_classes))
                )
        labels = None
        for data in outputs["labels"]:
            if labels is None:
                labels = torch.Tensor(data).view(-1)
            else:
                labels = torch.cat((labels, torch.Tensor(data).view(-1)))

        loss = self.val_loss(preds, labels)
        # save prediction and labels for to csv

        print(
            "calc loss epoch rank:",
            self.trainer.global_rank,
            "loss",
            loss,
            "best_val_loss",
            best_val_loss,
            preds.shape,
            labels.shape,
        )
        if best_val_loss is None or loss < best_val_loss:
            self.log(
                "best_val_loss",
                loss,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )
            self.best_val_acc_micro(preds, labels)
            self.best_val_acc_macro(preds, labels)
            self.best_val_f1(preds, labels)
            # log best val acc
            self.log(
                "best_val_acc_micro",
                self.best_val_acc_micro,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )
            self.log(
                "best_val_acc_macro",
                self.best_val_acc_macro,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )
            self.log(
                "best_val_f1",
                self.best_val_f1,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )
            if self.trainer.global_rank == 0 and isinstance(
                self.trainer.logger.experiment.id, str
            ):
                if not os.path.exists(self.trainer.default_root_dir):
                    os.makedirs(self.trainer.default_root_dir)
                # get class predictions
                preds = preds.argmax(dim=-1)
                pd.DataFrame(preds.cpu().numpy()).to_csv(
                    os.path.join(
                        self.trainer.default_root_dir,
                        self.trainer.logger.experiment.id + "_preds.csv",
                    ),
                    index=False,
                )
                pd.DataFrame(labels.cpu().numpy()).to_csv(
                    os.path.join(
                        self.trainer.default_root_dir,
                        self.trainer.logger.experiment.id + "_labels.csv",
                    ),
                    index=False,
                )
        self.validation_step_outputs.clear()
        self.validation_step_outputs = {"preds": [], "labels": []}
        print("done on_validation_epoch_end")
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        imgs, labels = batch[0], batch[1]

        preds = self.model(imgs)
        preds = preds.float()

        # calculate metrics
        self.test_acc_micro(preds, labels)
        self.test_acc_topk(preds, labels)
        self.test_f1(preds, labels)
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        final_result = []
        for i in range(preds.size(0)):
            row = {
                "id": ids[i],
                "preds": preds.data[i].cpu().numpy().tolist(),
                "labels": int(labels[i].cpu().numpy()),
                "chunk_nb": int(chunk_nb[i].cpu().numpy()),
                "split_nb": int(split_nb[i].cpu().numpy()),
            }
            final_result.append(row)

        df = pd.DataFrame(final_result)

        dest_path = os.path.join(
            self.trainer.default_root_dir, self.trainer.logger.experiment.id
        )
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        csv_file = os.path.join(dest_path, "test_results.csv")

        # If file does not exist, write with header
        if not os.path.isfile(csv_file):
            df.to_csv(csv_file, index=False)
        else:  # else it exists so append without writing the header
            df.to_csv(csv_file, mode="a", header=False, index=False)
        test_num_crop = self.trainer.datamodule.hparams.test_num_crop
        test_num_segments = self.trainer.datamodule.hparams.test_num_segment
        self.log(
            f"tns_{test_num_segments}_tnc_{test_num_crop}_test_acc_micro",
            self.test_acc_micro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"tns_{test_num_segments}_tnc_{test_num_crop}_test_acc_topk",
            self.test_acc_topk,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"tns_{test_num_segments}_tnc_{test_num_crop}_test_f1",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
