# %%
import os
import argparse
import cv2

# print number of cpu
print(os.cpu_count(), "cpus")
# print ram available
print(
    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024.0**3), "GB RAM"
)
# get params from command line

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="a_column_co_driver")
args = parser.parse_args()
folders = [
    args.folder,
]
folders = [
    "inner_mirror",
    # "kinect_color",
]  # 'a_column_driver', 'ceiling', 'inner_mirror', 'steering_wheel']
for folder in folders:
    video_folders = f"/home/user/Documents/datasets/driveact/{folder}/"
    dest_folder = f"/media/user/data/datasets/driveact/{folder}_frames/"
    # create dest folder if not exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    # convert each video in folder to frames
    for folder in os.listdir(video_folders):
        video_folder = os.path.join(video_folders, folder)
        for video in os.listdir(video_folder):
            if video.split(".")[-1].lower() != "mp4":
                continue
            video_path = os.path.join(video_folder, video)
            dest_path = os.path.join(dest_folder, folder, video.split(".")[0])
            print(video_path)
            print(dest_path)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
                # os.system(f'ffmpeg -i {video_path} {dest_path}/frame_%05d.png')
                # downscale to 480x270
            # get number of frames using opencv
            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(n_frames)
            # find if there are already frames
            n_frames_folder = len(os.listdir(dest_path))
            if n_frames_folder > 0:
                # delete 0 byte files
                os.system(f"find {dest_path} -size 0 -delete")
            n_frames_folder = len(os.listdir(dest_path))
            if n_frames_folder == n_frames:
                print("Already converted")
                continue
            os.system(
                f"ffmpeg -n -i {video_path} -vf scale=480:270 {dest_path}/frame_%05d.png"
            )


