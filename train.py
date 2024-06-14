"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""

# %%

import os
from argparse import ArgumentParser
import multiprocessing
import torchvision.transforms as transforms
import torchvision
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    stochastic_weight_avg,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from modules.base_module import BaseModule
from datamodule.base_datamod import BaseDataModule
from models.videomaev2 import VideoMaeV2

from datasets.driveact import DriveActDataset

from utils.video_transforms import VideoNormalize
from pytorchvideo.transforms import RandAugment
import torch
import json
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict

# import logging
# logging.basicConfig(
#     level=logging.DEBUG,
# )
# logging.getLogger("pytorch_lightning.trainer.trainer").setLevel(logging.DEBUG)
# logging.getLogger("pytorch_lightning.strategies.ddp").setLevel(logging.DEBUG)
import cv2

torch.set_num_threads(len(os.sched_getaffinity(0)))
cv2.setNumThreads(len(os.sched_getaffinity(0)))
num_cores = str(len(os.sched_getaffinity(0)))
os.environ["OMP_NUM_THREADS"] = num_cores
os.environ["MKL_NUM_THREADS"] = num_cores
os.environ["NUMEXPR_NUM_THREADS"] = num_cores
seed_everything(42)


DEBUGGING_FLAG = True


def to_normalized_float_tensor(vid):
    return vid.to(torch.float32) / 255


def main(hparams, network):
    project_folder = hparams.project_folder
    checkpoint_path = os.path.join(
        "./checkpoints/", hparams.model_name
    )  # "/opt/ml/checkpoints/"

    wandb_logger = WandbLogger(
        name=hparams.model_name,
        project=project_folder,
        # entity=os.getenv("WANDB_ENTITY"),
        offline=DEBUGGING_FLAG,
    )
    run_id = wandb_logger.experiment.id
    checkpoint_path = os.path.join(checkpoint_path, run_id)
    print(
        "Run ID: ",
        run_id,
        "Project: ",
        project_folder,
        "Name: ",
        hparams.model_name,
    )
    if hparams.gpus == -1:
        hparams.gpus = torch.cuda.device_count()
    num_cpu = len(os.sched_getaffinity(0))
    # if using slurm do not modify num_workers
    if "SLURM_JOB_ID" in os.environ:
        num_cpu = hparams.num_workers
    else:
        num_cpu = num_cpu if num_cpu < 8 * hparams.gpus else 16 * hparams.gpus
    hparams.num_workers = num_cpu

    print(hparams)

    model = BaseModule(network, hparams)
    if hparams.model_load_from_checkpoint:
        model = model.load_from_checkpoint(
            os.path.join(hparams.pretrained_folder, hparams.model_file), **vars(hparams)
        )

    wandb_logger.watch(model, log="all")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=hparams.early_stop_num,
            verbose=False,
            mode="min",
            check_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_path + "/",
            filename=hparams.model_name + "_" + run_id + "_{epoch:02d}-{val_loss:.3f}",
            save_last=True,
            save_top_k=4,
            monitor="val_loss",
            mode="min",
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
        # DeviceStatsMonitor(),
    ]
    weight_avg = True
    if weight_avg:
        callbacks.append(
            stochastic_weight_avg.StochasticWeightAveraging(
                swa_lrs=hparams.weight_avg_lr
            )
        )

    trainer = Trainer(
        accelerator="gpu",
        devices=hparams.gpus,
        max_epochs=hparams.max_nb_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        profiler="simple",
        benchmark=bool(hparams.fixed_data),
        precision=hparams.precision,
        default_root_dir=checkpoint_path,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams.accum_grad_batches,
        gradient_clip_val=hparams.gradient_clip_val,
        strategy="ddp",
        fast_dev_run=DEBUGGING_FLAG,
    )

    print("using video transforms with data augmentation")
    # totensor and crop to 224x224
    transforms_comp = transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224), antialias=True),
            to_normalized_float_tensor,
            VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_transforms = [
        torchvision.transforms.Resize((224, 224), antialias=True),
        to_normalized_float_tensor,
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if hparams.rand_aug:
        train_transforms.append(
            RandAugment(magnitude=hparams.rand_aug_M, num_layers=hparams.rand_aug_N),
        )
    train_transforms = transforms.Compose(train_transforms)

    dataset = DriveActDataset

    datamodule = BaseDataModule(
        dataset,
        transforms=transforms_comp,
        train_transforms=train_transforms,
        **vars(hparams),
    )
    print(f"Number of GPUs used: {trainer.device_ids}")
    trainer.fit(model, datamodule=datamodule)

    if hparams.test_model:
        # add test_num_crop to hparams
        test_num_crops = [1, 3, 5]
        test_num_segments = [1, 3, 5]
        # Find all checkpoint files
        ckpt_files = os.listdir(os.path.join(checkpoint_path))

        # Get files that match hparams.model_name + "_" + run_id
        ckpt_files = [
            run for run in ckpt_files if hparams.model_name + "_" + run_id in run
        ]

        # Create a dictionary that maps each validation loss to its corresponding file
        val_loss_to_file = {
            float(file.split("val_loss=")[-1].replace(".ckpt", "")): file
            for file in ckpt_files
            if "val_loss=" in file
        }

        # Get lowest validation loss
        lowest_val_loss = min(val_loss_to_file) if val_loss_to_file else None

        # Get the file with the lowest validation loss
        lowest_val_loss_file = val_loss_to_file.get(lowest_val_loss)

        print(f"Lowest validation loss: {lowest_val_loss}")
        print(f"File with lowest validation loss: {lowest_val_loss_file}")
        print(val_loss_to_file)
        n_strides = [-1, -2, 2, 3]
        for stride in n_strides:
            hparams.n_frames_stride = stride
            for tnc in test_num_crops:
                for tns in test_num_segments:
                    hparams.test_num_crop = tnc
                    hparams.test_num_segment = tns
                    datamodule = BaseDataModule(
                        dataset,
                        transforms=transforms_comp,
                        train_transforms=train_transforms,
                        **vars(hparams),
                    )
                    # if dataset is HMDB51 test last checkpoint
                    if hparams.dataset_artifact == "hmdb51-frames":
                        print("Testing last checkpoint")
                        trainer.test(
                            datamodule=datamodule,
                            ckpt_path="last",
                        )
                    else:
                        trainer.test(
                            datamodule=datamodule,
                            ckpt_path=os.path.join(
                                checkpoint_path, lowest_val_loss_file
                            ),
                        )
                    file_path = os.path.join(
                        checkpoint_path, run_id, "test_results.csv"
                    )
                    df = pd.read_csv(file_path)
                    # gropup by id

                    final_top1 = []
                    final_top5 = []
                    per_class_top1 = defaultdict(
                        lambda: [0, 0]
                    )  # [correct predictions, total predictions]
                    for id, group in df.groupby("id"):
                        feats = group["preds"].values
                        feats = np.array(feats)
                        # to float
                        feats = np.array(
                            [
                                np.fromstring(
                                    f.split("[")[1].split("]")[0], dtype=float, sep=","
                                )
                                for f in feats
                            ]
                        )
                        feats = np.mean(feats, axis=0)
                        pred = np.argmax(feats)
                        labels = group["labels"].values
                        label = labels[0]
                        top1 = (int(pred) == int(label)) * 1.0
                        top5 = (int(label) in np.argsort(-feats)[:5]) * 1.0
                        final_top1.append(top1)
                        final_top5.append(top5)
                        per_class_top1[label][
                            1
                        ] += 1  # increment total predictions for this class
                        if top1:
                            per_class_top1[label][
                                0
                            ] += 1  # increment correct predictions for this class

                    # calculate per class accuracy
                    per_class_accuracy = {
                        k: v[0] / v[1] for k, v in per_class_top1.items()
                    }
                    final_top1 = np.mean(np.array(final_top1))
                    final_top5 = np.mean(np.array(final_top5))
                    final_top1.shape
                    # calculate mean per class accuracy
                    per_class_accuracy = np.mean(list(per_class_accuracy.values()))
                    # update wandb
                    wandb_logger.experiment.log(
                        {
                            f"stride_{stride}_crop_{tnc}_seg_{tns}_test_top1": float(
                                final_top1
                            ),
                            f"stride_{stride}_crop_{tnc}_seg_{tns}_test_top5": float(
                                final_top5
                            ),
                            f"stride_{stride}_crop_{tnc}_seg_{tns}_per_class_accuracy": float(
                                per_class_accuracy
                            ),
                        }
                    )


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    # trainer args
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--early_stop_num", type=int, default=2)
    parser.add_argument(
        "--fixed-data",
        type=int,
        default=1,
        help="if 1, use fixed data can increase the speed of your system if your input sizes dont change.",
    )

    parser.add_argument("--rand_aug", type=int, default=1)
    parser.add_argument("--rand_aug_M", type=int, default=7)
    parser.add_argument("--rand_aug_N", type=int, default=4)
    parser.add_argument("--mixup", type=int, default=1)
    parser.add_argument("--warm_restarts", type=int, default=1)

    parser.add_argument("--gradient_clip_val", type=float, default=0.2)
    parser.add_argument("--model_load_from_checkpoint", type=int, default=0)

    # training args
    parser.add_argument("--max_nb_epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--accum_grad_batches", type=int, default=1)
    parser.add_argument("--t_max_scheduler", type=int, default=10)

    parser.add_argument("--lr", type=float, default=0.000008)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr_head", type=float, default=0.0005)
    parser.add_argument("--weight_decay_head", type=float, default=0.05)

    # train general params
    parser.add_argument("--weight_avg_lr", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # logging
    parser.add_argument("--project_folder", type=str, default="driveact")

    # model, view and dataset args
    parser.add_argument("--model_name", type=str, default="videomae-base")
    parser.add_argument("--dataset_artifact", type=str, default="driveact-frames")

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_frames", default=16, type=int)

    parser.add_argument("--test_model", dest="test_model", default=0, type=int)
    parser.add_argument("--yaml_file", type=str, default="")

    hparams, _ = parser.parse_known_args()
    dataset = DriveActDataset
    parser = dataset.add_model_specific_args(parser)
    # parse params
    hparams, _ = parser.parse_known_args()
    # load from yaml if exists
    if os.path.exists(hparams.yaml_file):
        with open(hparams.yaml_file) as f:
            yaml_params = yaml.load(f, Loader=yaml.FullLoader)
            for k, v in yaml_params.items():
                if "wandb" not in k and "transform" not in k and "dataset" not in k:
                    setattr(hparams, k, v["value"])
    print(hparams)
    network = VideoMaeV2
    print("using network", network)
    if not os.path.exists(hparams.yaml_file):
        # give the module a chance to add own params
        parser = network.add_model_specific_args(parser)
        # parse params
        hparams, _ = parser.parse_known_args()

    print(torch.cuda.device_count(), "GPUs", multiprocessing.cpu_count(), "CPUs")
    main(hparams, network)
