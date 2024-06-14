# %%
from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np
import torchvision
from argparse import ArgumentParser

DRIVEACT_DCT = {
    "closing_bottle": 0,
    "closing_door_inside": 1,
    "closing_door_outside": 2,
    "closing_laptop": 3,
    "drinking": 4,
    "eating": 5,
    "entering_car": 6,
    "exiting_car": 7,
    "fastening_seat_belt": 8,
    "fetching_an_object": 9,
    "interacting_with_phone": 10,
    "looking_or_moving_around (e.g. searching)": 11,
    "opening_backpack": 12,
    "opening_bottle": 13,
    "opening_door_inside": 14,
    "opening_door_outside": 15,
    "opening_laptop": 16,
    "placing_an_object": 17,
    "preparing_food": 18,
    "pressing_automation_button": 19,
    "putting_laptop_into_backpack": 20,
    "putting_on_jacket": 21,
    "putting_on_sunglasses": 22,
    "reading_magazine": 23,
    "reading_newspaper": 24,
    "sitting_still": 25,
    "taking_laptop_from_backpack": 26,
    "taking_off_jacket": 27,
    "taking_off_sunglasses": 28,
    "talking_on_phone": 29,
    "unfastening_seat_belt": 30,
    "using_multimedia_display": 31,
    "working_on_laptop": 32,
    "writing": 33,
}


class DriveActDataset(Dataset):
    def __init__(
        self,
        data_dir,
        set_type="train",
        task_type="midlevel",
        modal="kinect_color",
        fold="1",
        transforms=None,
        train_transforms=None,
        n_frames=8,
        **kwargs,
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            set_type (string): train, val, test
            task_type (string): midlevel, objectlevel, tasklevel
            modal (string): kinect_color, kinect_depth, kinect_ir, inner_mirror, a_column_co_driver, a_column_driver, ceiling,
                            steering_wheel
        """
        self.data_dir = data_dir
        self.set_type = set_type
        if self.set_type == "train":
            print("using train transforms", train_transforms)
            self.transforms = train_transforms
            self.convert_transforms = transforms
        else:
            self.transforms = transforms
        self.task_type = task_type
        self.modal = modal
        self.n_frames = n_frames
        self.n_frames_stride = kwargs.get("n_frames_stride", 1)
        self.h = 270
        self.w = 480
        self.h_adjust = self.h
        self.w_adjust = self.w
        split = set_type if set_type != "predict" else "test"
        self.data_df = pd.read_csv(
            os.path.join(
                data_dir,
                "activities_3s",
                modal,
                task_type + ".chunks_90.split_" + fold + "." + split + ".csv",
            )
        )
        self.data_df["activity"] = self.data_df["activity"].map(DRIVEACT_DCT)
        self.object_df = pd.read_csv(
            os.path.join(
                data_dir,
                "activities_3s",
                modal,
                "objectlevel" + ".chunks_90.split_" + fold + "." + split + ".csv",
            )
        )
        self.y = torch.tensor(self.data_df.activity.values, dtype=torch.long)
        if self.set_type == "test":
            self.test_num_segment = kwargs["test_num_segment"]
            self.test_num_crop = kwargs["test_num_crop"]
            self.short_side_size = kwargs["short_side_size"]
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.y)):
                        sample_label = self.y[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.data_df.iloc[idx])
                        self.test_seg.append((ck, cp))
            print(
                "test dataset",
                len(self.test_dataset),
                len(self.test_label_array),
                self.test_num_crop,
                self.test_num_segment,
            )
            self.length = len(self.test_dataset)
        else:
            self.length = len(self.data_df)
        print(
            self.length,
            set_type,
            "fold",
            fold,
            "num classes",
            len(torch.unique(self.y)),
            "num samples per class",
            torch.unique(self.y, return_counts=True),
            "task_type",
            task_type,
            "modal",
            modal,
            "n_frames",
            n_frames,
            "n_frames_stride",
            self.n_frames_stride,
        )

    def setup(self, stage=None):
        pass

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_dir", type=str, default="/media/ricardo/data/datasets/driveact"
        )
        parser.add_argument("--task_type", type=str, default="midlevel")
        parser.add_argument("--modal", type=str, default="inner_mirror")
        parser.add_argument("--num_classes", type=int, default=34)
        parser.add_argument("--fold", type=str, default="0")
        parser.add_argument("--short_side_size", type=int, default=224)

        return parser

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.set_type == "test":
            row = self.test_dataset[idx]
            chunk_nb, split_nb = self.test_seg[idx]
            frame_indices = list(range(row.frame_start, row.frame_end + 1))
            video_name = os.path.join(
                self.data_dir,
                self.modal + "_frames",
                row.file_id.split(".")[0],
            )
            buffer = self.read_all_frames(
                video_name,
                frame_indices,
            )

            video_name += f"_{frame_indices[0]}_{frame_indices[-1]}"
            if self.n_frames_stride == -1:
                # evenly sample 16 frames from buffer
                frame_indices = np.linspace(
                    0, buffer.shape[0] - 1, self.n_frames, dtype=int
                )
                buffer = buffer[frame_indices]
            elif self.n_frames_stride == -2:
                # randomly sample 16 frames from buffer
                frame_indices = np.random.choice(
                    np.arange(0, buffer.shape[0]),
                    self.n_frames,
                    replace=False,
                )
                frame_indices = np.sort(frame_indices)
                buffer = buffer[frame_indices]
            else:
                if self.test_num_crop > 1:
                    spatial_step = (
                        1.0
                        * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size)
                        / (self.test_num_crop - 1)
                    )
                else:
                    spatial_step = 0

                if self.test_num_segment > 1:
                    temporal_step = max(
                        1.0
                        * (buffer.shape[0] - self.n_frames)
                        / (self.test_num_segment - 1),
                        0,
                    )
                else:
                    temporal_step = 0

                if self.test_num_segment > 1:
                    temporal_start = int(chunk_nb * temporal_step)
                    spatial_start = int(split_nb * spatial_step)
                else:
                    # center crop
                    temporal_start = int((buffer.shape[0] - self.n_frames) / 2)
                    spatial_start = 0
                # check if the temporal stride fits in the buffer
                if (
                    temporal_start + self.n_frames * self.n_frames_stride
                    <= buffer.shape[0]
                ):
                    if buffer.shape[1] >= buffer.shape[2]:
                        buffer = buffer[
                            temporal_start : temporal_start
                            + self.n_frames
                            * self.n_frames_stride : self.n_frames_stride,
                            spatial_start : spatial_start + self.short_side_size,
                            :,
                            :,
                        ]
                    else:
                        buffer = buffer[
                            temporal_start : temporal_start
                            + self.n_frames
                            * self.n_frames_stride : self.n_frames_stride,
                            :,
                            spatial_start : spatial_start + self.short_side_size,
                            :,
                        ]
                else:
                    buffer = buffer[
                        temporal_start : temporal_start + self.n_frames,
                        :,
                        spatial_start : spatial_start + self.short_side_size,
                        :,
                    ]
            if self.transforms is not None:
                buffer = self.transforms(buffer)
            return buffer, self.test_label_array[idx], video_name, chunk_nb, split_nb
        else:
            row = self.data_df.iloc[idx]
            label = self.y[idx]
            frame_indices = list(range(row.frame_start, row.frame_end + 1))

            data = self.read_frame_folder(
                os.path.join(
                    self.data_dir, self.modal + "_frames", row.file_id.split(".")[0]
                ),
                frame_indices,
            )
            # read img folder
            if self.transforms is not None:
                data = self.transforms(data)
            return data, label

    def read_all_frames(self, frame_folder, frame_indices):
        if len(frame_indices) < self.n_frames:
            frame_indices = np.pad(
                frame_indices, (0, self.n_frames - len(frame_indices)), "edge"
            )
        frames = []
        for i in frame_indices:
            frame_path = os.path.join(frame_folder, f"frame_{i:05d}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frame_folder, f"frame_{i:05d}.jpg")
            # use torchvision to read image
            frame = torchvision.io.read_image(frame_path)
            frames.append(frame)
        frames = torch.stack(frames)
        return frames

    def read_frame_folder(self, frame_folder, frame_indices, r_sample=True):
        frame_indices = self.sample_pad_frames(frame_indices, r_sample=True)
        frames = []

        for i in frame_indices:
            frame_path = os.path.join(frame_folder, f"frame_{i:05d}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frame_folder, f"frame_{i:05d}.jpg")
            # use torchvision to read image
            frame = torchvision.io.read_image(frame_path)
            frames.append(frame)

        frames = torch.stack(frames)

        return frames

    def sample_pad_frames(self, frame_indices, r_sample=True):
        # evenly sample n frames from a list of frames
        if len(frame_indices) >= self.n_frames:
            start_frame = frame_indices[0]
            end_frame = frame_indices[-1]
            if self.set_type == "train" and r_sample:
                # randomly sample n frames
                frame_indices = np.random.choice(
                    np.arange(start_frame, end_frame + 1),
                    self.n_frames,
                    replace=False,
                )
                # sort the frames
                frame_indices = np.sort(frame_indices)
            else:
                # calculate the indices of the sampled frames
                frame_indices = np.linspace(
                    start_frame, end_frame, self.n_frames, dtype=int
                )
        else:
            # pad the frames with the last frame
            frame_indices = np.pad(
                frame_indices, (0, self.n_frames - len(frame_indices)), "edge"
            )
        return frame_indices.tolist()
