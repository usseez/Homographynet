# HomographyNET visualization

import time, numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from config import batch_size, num_workers, device
from data_gen import VideoHNDataset, DeepHNDataset
from mobilenet_v2 import MobileNetV2
from utils import AverageMeter
import cv2 as cv
import os
import pandas as pd

PATCH_SIZE = 256
RHO = 64
TOP_POINT  = (RHO, RHO)

SCALE_PRED = 1.0
SCALE_TGT = 1.0

OUT_DIR    = "./ORB_result_maxIters3000"
TEST_IMG_DIR = "./data/test2017"     # 원본 테스트 이미지 폴더
TEST_LIST    = "./data/test2017/test2017.txt"        # 한 줄에 한 파일명
FULL_SIZE    = (640, 480)               # 전처리 때 리사이즈한 크기(가로,세로)
video_path = './data/video/1_20180809_154805.mp4'


os.makedirs(OUT_DIR, exist_ok = True)

test_files = [l.strip() for l in open(TEST_LIST)]

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

def canonical_corners_rel(ps=PATCH_SIZE):
    # 패치 좌표계(0,0) 기준의 사각형
    return np.array([[0,0], [0,ps-1], [ps-1,ps-1], [ps-1,0]], dtype=np.float32)

def sanity_check_orb(Ip, Ip_p, H, out_path):
    w = cv.warpPerspective(Ip, H, (Ip_p.shape[1], Ip_p.shape[0]))
    cv.imwrite(out_path, np.hstack([to_uint8_img(Ip), to_uint8_img(Ip_p), w]))

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


def split_or_make_pair_from_channels(patch_tensor, H_gt_rel, ps=PATCH_SIZE):
    """
    입력 텐서에서 Ip, I′p 두 장을 얻는다.
    - (권장) C>=2이면 ch0=Ip, ch1=I′p 사용
    - 그 외엔 Ip에서 GT 호모그래피로 I′p를 생성
    """
    if isinstance(patch_tensor, torch.Tensor) and patch_tensor.dim() == 3:
        C, H, W = patch_tensor.shape
        if C >= 2:
            Ip    = to_uint8_img(patch_tensor[0:1, ...])  # (H,W) uint8 첫번째 채널
            Ip_p  = to_uint8_img(patch_tensor[1:2, ...])  # (H,W) uint8 두번째 채널
            return Ip, Ip_p
        else:
            Ip    = to_uint8_img(patch_tensor)
            Ip_p  = cv.warpPerspective(Ip, H_gt_rel, (ps, ps))
            return Ip, Ip_p
    else:
        Ip    = to_uint8_img(patch_tensor)
        Ip_p  = cv.warpPerspective(Ip, H_gt_rel, (ps, ps))
        return Ip, Ip_p

def orb_estimate_homography(Ip, Ip_p,
                            nfeatures=1500, ratio=0.8,
                            ransac_th=2.0, min_inliers=12):
    # 1) 입력 : uint8 GRAY
    if Ip.ndim == 3:
        Ip = cv.cvtColor(Ip, cv.COLOR_BGR2GRAY)
    if Ip_p.ndim == 3:
        Ip_p = cv.cvtColor(Ip_p, cv.COLOR_BGR2GRAY)
    if Ip.dtype != np.uint8:                                                    #type이 uint가 아니면, uint8로 변환
        Ip   = cv.normalize(Ip,   None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        Ip_p = cv.normalize(Ip_p, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # 2) ORB생성
    orb = cv.ORB_create(nfeatures=nfeatures, edgeThreshold=15, fastThreshold=7) #orb detector, descriptor 생성
    kp1, des1 = orb.detectAndCompute(Ip, None)                                  #각 이미지에서 kp, descriptor 추출
    kp2, des2 = orb.detectAndCompute(Ip_p, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, "No descriptors"

    # 3) KNN + Lowe ratio test
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)                        #브루트포스 매처?생성,,,,,
    knn = bf.knnMatch(des1, des2, k=2)                                          # des1의 각 descriptor에 대해 des2에서 가장 가까운 2개 매칭 찾음
    good = [m for m, n in knn if m.distance < ratio * n.distance]               # lowe ratio test 1등매칭거리m << 2등매칭거리n 여야 매칭으로 간주
    if len(good) < 4:                                                           # 4쌍 점 없으면 예외처리
        return None, f"Not enough matches ({len(good)})"

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)     # 매칭된 점 좌표를 (N, 1, 2)float32로 처리, #매칭된 keypoint를 호모그래피 계산에 맞는 형태로 뽑기
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 4) RANSAC + 인라이어 검사
    H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, ransac_th, confidence=0.995, maxIters=3000)  #pts1과 pts2사이 호모그래피 행렬 구함
    if H is None:
        return None, "Homography failed"
    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < min_inliers:
        return None, f"Too few inliers ({inliers})"                                                 #인라이어가 너무 적으면 호모그래피 구하기 실패로 간주

    # 5) 유효성 검사 (수치 폭주 방지)
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-8:
        return None, "Bad H (nan/inf)"
    s = np.linalg.svd(H[:2, :2], compute_uv=False)
    cond = (s[0] / max(s[1], 1e-12)) if s.size == 2 else 1.0
    if cond > 1e6:
        return None, "Ill-conditioned H"

    return H, None

def draw_boxes_on_image(img_gray, gt_xy, pred_xy, title, ace_px,
                        color_gt=(255,0,0), color_pred=(0,255,0)):
    vis = cv.cvtColor(to_uint8_img(img_gray), cv.COLOR_GRAY2BGR) if img_gray.ndim==2 else img_gray.copy()
    cv.polylines(vis, [gt_xy.astype(np.int32).reshape(-1,1,2)],  True, color_gt,   1, cv.LINE_AA)   #cv2::polylines 다각형 그리기 blue : gt
    cv.polylines(vis, [pred_xy.astype(np.int32).reshape(-1,1,2)], True, color_pred, 1, cv.LINE_AA)  #green : pred
    cv.putText(vis, title, (4, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
    cv.putText(vis, f"Mean Corner Error: {ace_px:.2f} px", (4, 55), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
    return vis

def mean_corner_error(dst_pred, dst_gt):
    return float(np.linalg.norm(dst_pred - dst_gt, axis=1).mean())

def rectify_and_draw(full_img_bgr, src_quad_full, base_quad_full, gt_quad_full, title):
    """
    full_img_bgr: 원본(BGR) 전체 이미지
    src_quad_full: '원본 좌표계'에서 직교화하고 싶은 사변형 (예: ORB 예측 사각형 or HNet 예측 사각형)
    base_quad_full: '원본 좌표계'에서 목표 사각형(정사각형, 보통 패치 위치의 base_full)
    gt_quad_full: '원본 좌표계'에서 GT 사각형
    title: 시각화 제목 문자열

    동작:
    - H_rect = getPerspectiveTransform(src_quad_full -> base_quad_full)
    - full_img를 H_rect로 warp → 직교화된 전체 이미지
    - gt_quad_full과 src_quad_full도 H_rect로 같이 좌표 변환 → 그 위에 파랑(GT), 초록(PRED) 박스 그리기
    """
    H_rect = cv.getPerspectiveTransform(
        src_quad_full.astype(np.float32),
        base_quad_full.astype(np.float32)
    )
    H, W = full_img_bgr.shape[:2]
    rectified = cv.warpPerspective(full_img_bgr, H_rect, (W, H))

    def warp_pts(quad):
        return cv.perspectiveTransform(
            quad.reshape(1,4,2).astype(np.float32), H_rect
        )[0]

    pred_rect = warp_pts(src_quad_full)   # 직교화된 프레임에서의 '추정' 사각형 (== base에 가깝게 나옴)
    gt_rect   = warp_pts(gt_quad_full)    # 직교화된 프레임에서의 GT 사각형
    ace       = mean_corner_error(pred_rect, gt_rect)

    vis = draw_boxes_on_image(rectified, gt_rect, pred_rect, title, ace)
    return vis


#엑셀에 누적 저장 함수
def append_feature_points_to_excel(
        image_id, base_full, dst_orb_full, pred_full, excel_path = "feature_points.xlsx", sheet_name="points"
):
    
    orb = np.asarray(dst_orb_full, dtype=np.float32)
    hnet = np.asarray(pred_full, dtype=np.float32)
    gt = np.asarray(base_full, dtype=np.float32)
    row = {
        "image_id" : image_id,

        "BASE_x0" : gt[0,0], "BASE_y0" : gt[0,1],
        "BASE_x1" : gt[1,0], "BASE_y1" : gt[1,1],
        "BASE_x2" : gt[2,0], "BASE_y2" : gt[2,1],
        "BASE_x3" : gt[3,0], "BASE_y3" : gt[3,1],

        "ORB_x0" : orb[0,0], "ORB_y0" : orb[0,1],
        "ORB_x1" : orb[1,0], "ORB_y1" : orb[1,1],
        "ORB_x2" : orb[2,0], "ORB_y2" : orb[2,1],
        "ORB_x3" : orb[3,0], "ORB_y3" : orb[3,1],

        "HNet_x0" : hnet[0,0], "HNet_y0" : hnet[0,1],
        "HNet_x1" : hnet[1,0], "HNet_y1" : hnet[1,1],
        "HNet_x2" : hnet[2,0], "HNet_y2" : hnet[2,1],
        "HNet_x3" : hnet[3,0], "HNet_y3" : hnet[3,1]
    }
    df_new = pd.DataFrame([row])

    if not os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_new.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists="overlay") as writer:
            try:
                existing = pd.read_excel(excel_path, sheet_name=sheet_name)
                startrow = len(existing) + 1
            except Exception:
                startrow = 0
            
            df_new.to_excel(writer, index=False, header=(startrow==0), sheet_name=sheet_name, startrow=startrow)


def frame_with_pred_gt(full_img_bgr, p_xy, v_pred8, v_gt8, ps=PATCH_SIZE, title='Pred vs GT'):
    base_rel = canonical_corners_rel(ps).astype(np.float32)
    gt_rel = base_rel + np.asarray(v_gt8, np.float32).reshape(4,2)
    pred_rel = base_rel + np.asarray(v_pred8, np.float32).reshape(4,2)

    p = np.array(p_xy, np.float32)
    gt_full = gt_rel + p
    pred_full = pred_rel + p
    base_full = base_rel + p

    title_left = 'Pred vs GT'
    title_right = 'Warped by H'
    ace = mean_corner_error(pred_full, gt_full)
    left = draw_boxes_on_image(full_img_bgr, gt_full, pred_full, title_left, ace)

    H_rect = cv.getPerspectiveTransform(pred_full.astype(np.float32), base_full.astype(np.float32))
    H, W = full_img_bgr.shape[:2]
    right = cv.warpPerspective(full_img_bgr, H_rect, (W,H))
    right = draw_boxes_on_image(right, base_full, base_full, title_right, ace)
    # cv.putText(right, title_right, (4,25), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2,cv.LINE_AA)

    vis = np.concatenate([left, right], axis=1)

    return vis






def prediction(filename):
    # load model
    print(f'loading {filename}...')
    model = MobileNetV2()   #모델 구조 정의
    chpt= torch.load(filename, map_location=device)                     #가중치 파일 읽기
    model.load_state_dict(chpt, strict=False)                           # 모델 구조에 가중치 매핑

    model.eval()
    model.to(device)




    # image preprocessing
    if video_path is not None:
        test_dataset = VideoHNDataset(video_path, FULL_SIZE, RHO, PATCH_SIZE, 128, [(RHO,RHO)], 1, None)    
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)                              # 동영상 넣기
        is_video_mode = True
    else:
        test_dataset = DeepHNDataset('test')
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)
        is_video_mode = False


    num_samples = len(test_dataset)


    os.makedirs("./video_out", exist_ok=True)
    video_path_out = "./video_out/pred_vs_gt3.mp4"
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = None
    fps = 15


    # Loss functionnn.MSELoss()
    criterion = nn.MSELoss().to(device)
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
    global_idx = 0
    batch_idx = 0
    for batch in tqdm(test_loader):

        if is_video_mode:
            img, target, meta = batch
        else:
            img, target = batch
            meta = None
        
        # Move to CPU, if available
        img = img.to(device)  # [N, 3, 128, 128]
        target = target.float().to(device)  # [N, 8]

        # Forward prop.
        with torch.inference_mode():
            start = time.time()
            out = model(img)  # [N, 8]  예측한 사변형 코너좌표
            end = time.time()
            elapsed = elapsed + (end - start)

        out_scaled = out * 2.0
        loss_meter.update(criterion(out_scaled, target).item(), img.size(0))

        out_np = out_scaled.detach().cpu().numpy()  #gpu에 있는 텐서를 cpu로 복사
        tgt_np = target.detach().cpu().numpy()

        for i in range(out_np.shape[0]): #out_np.shape[0] 배치크기만큼을 반복하며
            v_pred = out_np[i]  
            v_gt = tgt_np[i]
            name_prefix = f"b{batch_idx}_i{i}"

            p_xy = (RHO, RHO)

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



            #visualization full image
            if is_video_mode:
                frame_id = int(meta["frame_id"][i].item() if torch.is_tensor(meta["frame_id"][i]) else meta["frame_id"][i])
                full_img_bgr = cv.cvtColor(test_dataset.frames[frame_id], cv.COLOR_GRAY2BGR)
            else:
                img_rel_path = test_files[global_idx]
                full_img_bgr = cv.imread(os.path.join(TEST_IMG_DIR, img_rel_path), cv.IMREAD_COLOR)
                full_img_bgr = cv.resize(full_img_bgr, FULL_SIZE)

            frame_vis = frame_with_pred_gt(full_img_bgr, p_xy, v_pred, v_gt, ps = PATCH_SIZE)

            if writer is None:
                H, W = frame_vis.shape[:2]
                writer = cv.VideoWriter(video_path_out, fourcc, 15, (W,H))

            writer.write(frame_vis)

            
            global_idx += 1


    


        batch_idx += 1

    if writer is not None:
        writer.release()
        print(f"[INFO] Saved video to : {video_path_out}")


    print('Elapsed per sample: {:.5f} ms'.format(elapsed / max(1, num_samples) * 1000))
    print('L2 Loss (out vs target): {:.4f}'.format(loss_meter.avg))

    print('===== Equality check (scale-invariant) ======')
    print('mean relative : {:.6e}'.format(diff_meter.avg))     #||H_pred − α H_gt||_F / ||H_pred||_F
    print('Reprojection error : mean {:.4f}, max {:.4f}'.format(
        reproj_mean_meter.avg, reproj_max_meter.avg))           #dst_gt (px)

    print('****Sanity: getPerspectiveTransform vs findHomography (lower≈better)****')
    print('pred pair diff: {:.6e}   gt pair diff: {:.6e}'.format(
        cv_consistency_pred.avg, cv_consistency_gt.avg))

    print('****', out_scaled[0].min().item(), out_scaled[0].max().item(),
      target[0].min().item(), target[0].max().item(), '****')


if __name__ == '__main__':
    filename = './weights/homonet.pt'
    device = torch.device(device)
    prediction(filename)



