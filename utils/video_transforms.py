import torchvision.transforms.v2.functional as F

class VideoNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        T, C, H, W = vid.shape
        vid = vid.view(-1, C, H, W)
        vid = F.normalize(vid, mean=self.mean, std=self.std)
        vid = vid.view(T, C, H, W)
        return vid
