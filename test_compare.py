
import time, numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from config import batch_size, num_workers, device
from data_gen import DeepHNDataset
from mobilenet_v2 import MobileNetV2
from utils import AverageMeter
import cv2 as cv
import os

PATCH_SIZE = 256
RHO = 64
TOP_POINT  = (RHO, RHO)

SCALE_PRED = 1.0
SCALE_TGT = 1.0

OUT_DIR    = "./warp_results"
os.makedirs(OUT_DIR, exist_ok = True)


def canonical_corners(ps = PATCH_SIZE):  #4 꼭짓점좌표를 고정된 순서로 반환
    return np.array([[0,0], [0,ps-1], [ps-1,ps-1], [ps-1,0]], dtype=np.float32)

def vec8_to_corners(vec8, ps, scale):   # 모델 8차원 오프셋을 4개 코너좌표로 변환
    base = canonical_corners(ps)
    off = (np.asarray(vec8).reshape(4,2) * float(scale)).astype(np.float32)
    return base + off

def H_from_vec8(vec8, ps, scale):   # 위에서 만든 점으로 H 구하기
    src = canonical_corners(ps)
    dst = vec8_to_corners(vec8, ps, scale)
    return cv.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))       #perspectiveTransform 행렬


def normalize_H(H): # H 정규화
    H = H.astype(np.float64)
    s = H[2, 2] if abs(H[2,2]) > 1e-12 else np.sign(H[2,2] if H[2,2] != 0 else 1.0)
    return H / s


def scale_invariant_diff(H1, H2):    #H1, H2 의 근접도 비교
    h1 = H1.reshape(-1)
    h2 = H2.reshape(-1)
    denom = float(np.dot(h2, h2)) if float(np.dot(h2, h2)) != 0 else 1.0
    alpha = float(np.dot(h2, h1)) / denom
    num = np.linalg.norm(H1 - alpha*H2, ord='fro')      #numpy 벡터 정규화
    den = np.linalg.norm(H1, ord = 'fro')
    return (num / (den if den>0 else 1.0)), alpha


def reprojection_err_px(H, dst_gt, ps): #픽셀 오차 계산
    src = canonical_corners(ps).reshape(-1,1,2).astype(np.float32)
    proj = cv.perspectiveTransform(src, H).reshape(-1,2)
    e = np.linalg.norm(proj - dst_gt, axis=1)
    return float(e.mean()), float(e.max())


def base_corners_from_top(top_point = TOP_POINT, ps = PATCH_SIZE):
    x, y = top_point
    TL = (x, y)
    BL = (x, y + ps)
    BR = (x + ps, y + ps)
    TR = (x + ps, y)
    return np.array([TL, BL, BR, TR], dtype=np.float32)


def canonical_corners_rel(ps=PATCH_SIZE):
    # 패치 좌표계(0,0) 기준의 사각형
    return np.array([[0,0], [0,ps-1], [ps-1,ps-1], [ps-1,0]], dtype=np.float32)


def to_uint8_img(tensor_or_np): #gray scale 이미지를 저장/보기용 uint8로 바꾸기
    """ (C,H,W) torch or (H,W)/(H,W,3) np → uint8 BGR/GRAY """
    if isinstance(tensor_or_np, torch.Tensor):          # tensor_or_np가 torch.Tensor 타입이면
        x = tensor_or_np.detach().cpu().float() #텐서를 cpu로 float형태로 저장
        if x.dim() == 3:    #3차원 tensor면
            C, H, W = x.shape
            if C == 1:  # gray scale일 때
                img = x[0].numpy()
                m, M = float(img.min()), float(img.max())
                if M > m: img = (img - m) / (M - m)
                return (img * 255).clip(0,255).astype(np.uint8)
            else:
                img = x[:3].permute(1,2,0).numpy()  # HWC
                m, M = float(img.min()), float(img.max())
                if M > m: img = (img - m) / (M - m)
                img = (img * 255).clip(0,255).astype(np.uint8)
                # tensor는 보통 RGB; OpenCV 저장은 BGR이라면 cv.cvtColor로 바꿀 수 있음
                return img
        elif x.dim() == 2:  #2차원 tesnsor면
            img = x.numpy()
            m, M = float(img.min()), float(img.max())
            if M > m: img = (img - m) / (M - m)
            return (img * 255).clip(0,255).astype(np.uint8)
        else:
            raise ValueError("Unsupported tensor shape")
    else:
        img = tensor_or_np
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            m, M = float(img.min()), float(img.max())
            if M > m: img = (img - m) / (M - m)
            img = (img * 255).clip(0,255).astype(np.uint8)
        return img


def save_warp_results_patch(patch_tensor, v_pred8, v_gt8, name_prefix,
                            scale_pred=1.0, scale_gt=1.0, ps=PATCH_SIZE):       #모델 입력 패치 -> v_pred8, v_gt8로 warpPerspoecive를 돌려 사변형으로 변형해 저장
    """
    patch_tensor : 모델 입력 패치 (C,H,W) torch.Tensor 또는 (H,W), (H,W,3) np
    v_pred8, v_gt8 : 길이 8 오프셋
    """
    patch = to_uint8_img(patch_tensor)  # (H,W) 또는 (H,W,3) patch를 저장하기 위해 uint8로 변환
    base_rel = canonical_corners_rel(ps)  # [TL,BL,BR,TR] in patch coords 원본 patch 좌표

    pred_rel = base_rel + np.asarray(v_pred8, dtype=np.float32).reshape(4,2) * float(scale_pred)    #사변형의 실제 좌표 만들기
    gt_rel   = base_rel + np.asarray(v_gt8,   dtype=np.float32).reshape(4,2) * float(scale_gt)

    H_pred_rel = cv.getPerspectiveTransform(base_rel, pred_rel)
    H_gt_rel   = cv.getPerspectiveTransform(base_rel, gt_rel)

    # H_pred_rel_inv = np.linalg.inv(H_pred_rel)
    # H_gt_rel_inv = np.linalg.inv(H_gt_rel)

    patch_pred = cv.warpPerspective(patch, H_pred_rel, (ps, ps)) #input : patch, H_pred_rel호모그래피로 (ps,ps)크기의 output : patch_pred 생성
    patch_gt   = cv.warpPerspective(patch, H_gt_rel,   (ps, ps))

    # cv.imwrite(os.path.join(OUT_DIR, f"{name_prefix}_pred_patch.png"), patch_pred)
    # cv.imwrite(os.path.join(OUT_DIR, f"{name_prefix}_gt_patch.png"),   patch_gt)
    side = np.concatenate([patch_gt if patch_gt.ndim==2 else cv.cvtColor(patch_gt, cv.COLOR_BGR2GRAY),
                           patch_pred if patch_pred.ndim==2 else cv.cvtColor(patch_pred, cv.COLOR_BGR2GRAY)], axis=1)
    cv.imwrite(os.path.join(OUT_DIR, f"{name_prefix}_gt_vs_pred_patch.png"), side)


def prediction(filename):


    # load model
    print(f'loading {filename}...')
    model = MobileNetV2()   #모델 구조 정의
    chpt= torch.load(filename, map_location=device) #가중치 파일 읽기
    model.load_state_dict(chpt, strict=False)   # 모델 구조에 가중치 매핑

    model.eval()
    model.to(device)


    # image preprocessing
    test_dataset = DeepHNDataset('test')    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)

    num_samples = len(test_dataset)

    # Loss functionnn.MSELoss()
    criterion = nn.L1Loss().to(device)
    loss_meter = AverageMeter()

    #지표 누적
    diff_meter = AverageMeter()
    reproj_mean_meter = AverageMeter()
    reproj_max_meter = AverageMeter()
    cv_consistency_pred = AverageMeter()
    cv_consistency_gt = AverageMeter()




    elapsed = 0.0
    src_corners = canonical_corners(PATCH_SIZE).astype(np.float32)


    # Batches
    batch_idx = 0
    for (img, target) in tqdm(test_loader):
        # Move to CPU, if available
        # img = F.interpolate(img, size=(img.size(2) // 2, img.size(3) // 2), mode='bicubic', align_corners=False)
        img = img.to(device)  # [N, 3, 128, 128]
        target = target.float().to(device)  # [N, 8]

        # Forward prop.
        with torch.no_grad():
            start = time.time()
            out = model(img)  # [N, 8]  예측한 사변형 코너좌표
            end = time.time()
            elapsed = elapsed + (end - start)

        out_scaled = out * 2.0
        loss_meter.update(criterion(out_scaled, target).item(), img.size(0))

        out_np = out_scaled.detach().cpu().numpy()  #gpu에 있는 텐서를 cpu로 복사
        tgt_np = target.detach().cpu().numpy()
        # print(out_np)
        # break
        for i in range(out_np.shape[0]): #out_np.shape[0] 배치크기만큼을 반복하며
            v_pred = out_np[i]  
            v_gt = tgt_np[i]
            name_prefix = f"b{batch_idx}_i{i}"

            #Homography based on prediction
            H_pred = H_from_vec8(v_pred, PATCH_SIZE, SCALE_PRED)
            dst_pred = vec8_to_corners(v_pred, PATCH_SIZE, SCALE_PRED).astype(np.float32)
            H_pred_cv, _ = cv.findHomography(src_corners, dst_pred, method=0)
            
            #GT
            H_gt = H_from_vec8(v_gt, PATCH_SIZE, SCALE_TGT)
            dst_gt = vec8_to_corners(v_gt, PATCH_SIZE, SCALE_TGT).astype(np.float32)
            H_gt_cv, _ = cv.findHomography(src_corners, dst_gt, method=0)


            #normalization
            Hp = normalize_H(H_pred)    # H33 = 1로 normalize
            Hg = normalize_H(H_gt)
            Hp_c = normalize_H(H_pred_cv) if H_pred_cv is not None else Hp
            Hg_c = normalize_H(H_gt_cv) if H_gt_cv is not None else Hg

            #H vs GT Homography compare
            diff, alpha = scale_invariant_diff(Hp, Hg)
            diff_meter.update(diff, 1)

            # (2) 재투영 오차 (H_pred로 src→dst_gt)
            mean_px, max_px = reprojection_err_px(H_pred, dst_gt, PATCH_SIZE)
            reproj_mean_meter.update(mean_px, 1)
            reproj_max_meter.update(max_px, 1)

            # (3) 참고: getPerspectiveTransform vs findHomography 일치도
            diff_pred_cv, _ = scale_invariant_diff(Hp, Hp_c)
            diff_gt_cv,   _ = scale_invariant_diff(Hg, Hg_c)
            cv_consistency_pred.update(diff_pred_cv, 1)
            cv_consistency_gt.update(diff_gt_cv, 1)

            save_warp_results_patch(img[i], v_pred, v_gt, name_prefix, scale_pred = 1.0, scale_gt=1.0)
        batch_idx += 1

    print('Elapsed per sample: {:.5f} ms'.format(elapsed / max(1, num_samples) * 1000))
    print('L1 Loss (out vs target): {:.4f}'.format(loss_meter.avg))

    print('== Equality check (scale-invariant) ==')
    print('mean relative : {:.6e}'.format(diff_meter.avg))     #||H_pred − α H_gt||_F / ||H_pred||_F
    print('Reprojection error : mean {:.4f}, max {:.4f}'.format(
        reproj_mean_meter.avg, reproj_max_meter.avg))           #dst_gt (px)

    print('Sanity: getPerspectiveTransform vs findHomography (lower≈better)')
    print('pred pair diff: {:.6e}   gt pair diff: {:.6e}'.format(
        cv_consistency_pred.avg, cv_consistency_gt.avg))

    print(out_scaled[0].min().item(), out_scaled[0].max().item(),
      target[0].min().item(), target[0].max().item())


if __name__ == '__main__':
    filename = './weights/homonet.pt'
    device = torch.device(device)
    prediction(filename)