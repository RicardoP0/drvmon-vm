# %%
import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger

from modules.base_module import BaseModule
from modules.multi_view_module import MVModule
from datamodule.base_datamod import BaseDataModule
from models.swin_transformer import SwimTransformer
from models.evad_hm_stm import EVADSTM

from modules.heatmap_module import HeatmapModule
from models.videomaev2 import VideoMaeV2
from models.videomaev2_heatmap import VideoMaeV2Heatmap

import torch
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

seed_everything(42)


def to_normalized_float_tensor(vid):
    return vid.to(torch.float32) / 255


def main(hparams, network):
    if hparams.gpus == -1:
        hparams.gpus = torch.cuda.device_count()
    num_cpu = len(os.sched_getaffinity(0))
    # if using slurm do not modify num_workers
    if "SLURM_JOB_ID" in os.environ:
        num_cpu = hparams.num_workers
    else:
        num_cpu = num_cpu if num_cpu < 8 * hparams.gpus else 8 * hparams.gpus
    hparams.num_workers = num_cpu

    print(hparams)

    # init module
    # if "multi" in hparams.model_name:
    #     model = MVModule(network, hparams)
    # else:
    #     model = BaseModule(network, hparams)
    checkpoint = torch.load(
        hparams.model_file, map_location=lambda storage, loc: storage
    )
    file_hparams = dict(checkpoint["datamodule_hyper_parameters"])
    # convert namespace to dict
    hparams = vars(hparams)
    # overwrite file hparams with command line hparams Namespace
    file_hparams.update(hparams)
    hparams.update(file_hparams)
    # convert back to namespace
    hparams = argparse.Namespace(**hparams)

    if "multi" in hparams.model_name:
        model = MVModule.load_from_checkpoint(
            hparams.model_file, model=network, **vars(hparams)
        )
    elif "hm" in hparams.model_name:
        model = HeatmapModule.load_from_checkpoint(
            hparams.model_file, model=network, **vars(hparams)
        )
    else:
        model = BaseModule.load_from_checkpoint(
            hparams.model_file, model=network, **vars(hparams)
        )
    # model.model = torch.compile(
    #     model.model, disable=DEBUGGING_FLAG, mode="reduce-overhead"
    # )
    project_folder = hparams.project_folder
    checkpoint_path = os.path.join(
        "./checkpoints/", hparams.model_name
    )  # "/opt/ml/checkpoints/"

    wandb_logger = WandbLogger(
        name=hparams.model_name,
        project=project_folder,
        # entity=os.getenv("WANDB_ENTITY"),
    )
    run_id = wandb_logger.experiment.id
    print(
        "Run ID: ",
        run_id,
        "Project: ",
        project_folder,
        "Name: ",
        hparams.model_name,
    )
    wandb_logger.watch(model, log="all")
    # wandb_logger.experiment.use_artifact(hparams.dataset_artifact + ":latest")

    trainer = Trainer(
        accelerator="gpu",
        devices=hparams.gpus,
        max_epochs=hparams.max_nb_epochs,
        logger=wandb_logger,
        profiler="simple",
        # deterministic=False,
        benchmark=bool(hparams.fixed_data),
        precision=hparams.precision,
        default_root_dir=checkpoint_path,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams.accum_grad_batches,
        strategy="ddp",
        # limit_train_batches=0.5,
        # limit_val_batches=0.5,
        num_sanity_val_steps=0,
    )

    datamodule = BaseDataModule(
        **vars(hparams),
    )
    datamodule.setup("test")
    print(f"Number of GPUs used: {trainer.device_ids}")
    # trainer.fit(model, datamodule=datamodule)
    model.eval()
    model.freeze()
    trainer.test(model, datamodule=datamodule)
    file_path = os.path.join(checkpoint_path, run_id, "test_results.csv")
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
                np.fromstring(f.split("[")[1].split("]")[0], dtype=float, sep=",")
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
        per_class_top1[label][1] += 1  # increment total predictions for this class
        if top1:
            per_class_top1[label][
                0
            ] += 1  # increment correct predictions for this class

    # calculate per class accuracy
    per_class_accuracy = {k: v[0] / v[1] for k, v in per_class_top1.items()}

    final_top1 = np.mean(np.array(final_top1))
    final_top5 = np.mean(np.array(final_top5))
    final_top1.shape
    # Top1: 0.8686274509803922 Top5: 0.9790849673202614
    # calculate mean per class accuracy
    per_class_accuracy = np.mean(list(per_class_accuracy.values()))
    tnc = hparams.test_num_crop
    tns = hparams.test_num_segment
    # update wandb
    wandb_logger.experiment.log(
        {
            f"c_{tnc}_s_{tns}_test_top1": float(final_top1),
            f"c_{tnc}_s_{tns}_test_top5": float(final_top5),
            f"c_{tnc}_s_{tns}_per_class_accuracy": float(per_class_accuracy),
        }
    )
    # load best model
    # if trainer.is_global_zero and hparams.gpus == 1 and weight_avg == False:
    #     trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    # trainer args
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--test_num_crop", default=5, type=int)
    parser.add_argument("--test_num_segment", default=5, type=int)
    parser.add_argument("--target_kp_loss_weight", type=int, default=0)
    # parser.add_argument("--merge_type", type=str, default="sim")
    parser.add_argument("--n_frames_stride", type=int, default=1)
    parser.add_argument("--short_side_size", type=int, default=224)

    # logging
    # parser.add_argument("--project_folder", type=str, default="hmdb51")

    # model, view and dataset args
    parser.add_argument("--model_name", type=str, default="evad-hm-stm")
    # parser.add_argument("--dataset_artifact", type=str, default="driveact-frames")
    parser.add_argument(
        "--model_file",
        type=str,
        # default="/media/ricardo/data2/driveact_models/results/drive/topk+merge/evad-hm-stm_57f13wlm_fold0.ckpt",
        default="/media/ricardo/data2/driveact_models/results/drive_ablation/evad-hm-stm_eh9t1ovh_epoch=82-val_loss=0.582.ckpt",
    )

    # data args
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get(
            "DATA_INPUT_DIR", "/media/ricardo/data/datasets/driveact"
        ),
        help="path to the data",
    )
    # get num of cpu cores to use as default for num_workers

    parser.add_argument("--num_workers", type=int, default=8)

    # parser.add_argument("--num_classes_obj", default=0, type=int)  # 17
    # parser.add_argument("--object_cls", default=0, type=int)
    hparams, _ = parser.parse_known_args()
    if hparams.model_name == "videoswim-base":
        network = SwimTransformer
    elif hparams.model_name == "videomae-base":
        network = VideoMaeV2
    elif hparams.model_name == "videomae-hm":
        network = VideoMaeV2Heatmap
    elif hparams.model_name == "evad-hm-stm":
        network = EVADSTM
    # parse params
    hparams, _ = parser.parse_known_args()

    main(hparams, network)
