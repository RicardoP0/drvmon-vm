import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from datamodule.mixup import Mixup
import torch


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, dataset, **kwargs):
        super().__init__()
        # self.__dict__.update(kwargs)
        self.save_hyperparameters()
        self.dataset = dataset

    def prepare_data(self):
        pass

    def _one_hot(self, x, num_classes, smoothing=0.0):
        off_value = smoothing / num_classes
        on_value = 1.0 - smoothing + off_value
        x = x.long().view(-1, 1)
        return torch.full(
            (x.size()[0], num_classes), off_value, device=x.device
        ).scatter_(1, x, on_value)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = self.dataset(
                set_type="train",
                **self.hparams,
            )
            self.train_dataset.setup()
            self.val_dataset = self.dataset(
                set_type="val",
                **self.hparams,
            )
            self.val_dataset.setup()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = self.dataset(
                set_type="test",
                **self.hparams,
            )

        if stage == "predict":
            pass

    def train_dataloader(self):
        # # Balanced batch sampler
        # loss_weights = self.train_dataset.calc_class_weights()
        # weight_sample = torch.tensor([loss_weights[i] for i in self.train_dataset.y])
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_sample, len(self.train_dataset))
        if self.hparams.mixup:
            mixup_fn = Mixup(
                mixup_alpha=0.8,
                cutmix_alpha=1,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=self.hparams.label_smoothing,
                num_classes=self.hparams.num_classes,
            )

            def collate_fn(batch):
                x, y = default_collate(batch)
                B, C, T, H, W = x.shape
                x = x.view(B, C * T, H, W)
                if isinstance(y, list):
                    y_tmp = y[0]
                    y_rest = y[1:]
                    x, y_res = mixup_fn(x, y_tmp)
                    y = [y_res] + y_rest
                    x = x.view(B, C, T, H, W)
                    return x, y
                x, y = mixup_fn(x, y)
                x = x.view(B, C, T, H, W)
                return x, y

            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                # prefetch_factor=1,
                persistent_workers=True,
                collate_fn=collate_fn,
                drop_last=True,
                # sampler=sampler,
                # worker_init_fn=dataload_init
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            # prefetch_factor=1,
            persistent_workers=True,
            # worker_init_fn=dataload_init
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        pass
