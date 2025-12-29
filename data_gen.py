import pickle

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from config import im_size
import os
class VideoHNDataset(Dataset):
    def __init__(self, video_path,
                 size=(640, 480),      # FULL_SIZE와 동일
                 rho=64,
                 patch_size=256,
                 im_size=128,          # 모델 입력 해상도(기존 im_size와 맞춰주세요)
                 positions=None,       # 패치 좌상단 좌표들
                 stride=1,             # 프레임 샘플링 간격
                 max_frames=None,      # 최대 프레임 수 제한(옵션)
                 seed=0):
        self.video_path = video_path
        self.size = size
        self.rho = int(rho)
        self.patch_size = int(patch_size)
        self.im_size = int(im_size)
        self.stride = int(stride)
        self.rng = np.random.RandomState(seed)

        # 패치 위치(테스트 기본: (rho, rho) 한 곳)
        if positions is None:
            self.positions = [(self.rho, self.rho)]
        else:
            self.positions = list(positions)

        # 동영상 읽어서 프레임 회수/인덱스 목록 구성
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.frames = []   # 메모리에 적재(간단/안전)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.stride != 0:
                idx += 1
                continue
            frame = cv.resize(frame, self.size)
            gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.frames.append(gray)
            idx += 1
            if (max_frames is not None) and (len(self.frames) >= max_frames):
                break
        cap.release()

        # 각 프레임마다 positions 개수만큼 샘플 생성
        self.index = []  # (frame_id, pos_id)
        for f_id in range(len(self.frames)):
            for p_id in range(len(self.positions)):
                self.index.append((f_id, p_id))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        frame_id, pos_id = self.index[i]
        img = self.frames[frame_id]                    # (H,W) GRAY, FULL_SIZE
        top = self.positions[pos_id]                  # (x,y) = (col,row)
        TL = (top[0], top[1])
        BL = (top[0], top[1] + self.patch_size)
        BR = (top[0] + self.patch_size, top[1] + self.patch_size)
        TR = (top[0] + self.patch_size, top[1])
        four_points = np.float32([TL, BL, BR, TR])

        # perturbed 4점(= GT)
        jitter = self.rng.randint(-self.rho, self.rho + 1, size=(4, 2)).astype(np.float32)
        perturbed_four_points = four_points + jitter

        # H_ab (patch→perturbed) 구하고, H_ba로 전체 이미지를 warp
        H = cv.getPerspectiveTransform(four_points, perturbed_four_points)
        H_inv = np.linalg.inv(H)
        warped_full = cv.warpPerspective(img, H_inv, self.size)

        # 패치 잘라서 Ip / Ip′ 생성
        x0, y0 = TL
        x1, y1 = BR
        Ip  = img[y0:y1, x0:x1]             # (patch_size, patch_size)
        Ipp = warped_full[y0:y1, x0:x1]     # (patch_size, patch_size)

        # 모델 입력 형태로 리사이즈 & 채널 쌓기
        Ip_r  = cv.resize(Ip,  (self.im_size, self.im_size))
        Ipp_r = cv.resize(Ipp, (self.im_size, self.im_size))
        img3  = np.zeros((self.im_size, self.im_size, 3), np.float32)
        img3[..., 0] = Ip_r / 255.0
        img3[..., 1] = Ipp_r / 255.0
        # img3[..., 2] = 0 (기존과 동일)

        # 타깃: 8D offset (perturbed - base)
        target = (perturbed_four_points - four_points).reshape(8).astype(np.float32)

        # 메타(엑셀 저장용 id)
        meta = {
            "frame_id": frame_id,
            "pos_id": pos_id,
            "image_id": f"{os.path.basename(self.video_path)}#f{frame_id}_p{pos_id}",
            "top_point": (int(top[0]), int(top[1])),
        }

        # (C,H,W)
        img_tensor = np.transpose(img3, (2, 0, 1)).astype(np.float32)
        return img_tensor, target, meta



class DeepHNDataset(Dataset):
    def __init__(self, split):
        filename = '/home/ubuntu/Works/CALIBRATION/HomographyNet/data/{}.pkl'.format(split)
        print('loading {}...'.format(filename))
        with open(filename, 'rb') as file:
            samples = pickle.load(file)
        np.random.shuffle(samples)
        self.split = split
        self.samples = samples

    def __getitem__(self, i):
        sample = self.samples[i]
        image, four_points, perturbed_four_points = sample
        img0 = image[:, :, 0]
        img0 = cv.resize(img0, (im_size, im_size))
        img1 = image[:, :, 1]
        img1 = cv.resize(img1, (im_size, im_size))
        img = np.zeros((im_size, im_size, 3), np.float32)
        img[:, :, 0] = img0 / 255.
        img[:, :, 1] = img1 / 255.
        img = np.transpose(img, (2, 0, 1))  # HxWxC array to CxHxW
        H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
        target = np.reshape(H_four_points, (8,))
        return img, target

    def __len__(self):
        return len(self.samples)


# if __name__ == "__main__":
#     train = DeepHNDataset('train')
#     print('num_train: ' + str(len(train)))
#     valid = DeepHNDataset('valid')
#     print('num_valid: ' + str(len(valid)))

#     print(train[0])
#     print(valid[0])
