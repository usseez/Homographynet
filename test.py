import time

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from config import batch_size, num_workers, device
from data_gen import DeepHNDataset
from mobilenet_v2 import MobileNetV2
from utils import AverageMeter


SCALE_PRED = 2.0

device = torch.device(device)

if __name__ == '__main__':
    filename = './weights/homonet.pt'

    print('loading {}...'.format(filename))
    model = MobileNetV2()
    model.load_state_dict(torch.load(filename))
    model.to(device).eval()

    pin = (device.type == 'cuda')   #gpu 쓰면 true 아니면 false
    test_dataset = DeepHNDataset('test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, drop_last=False)

    num_samples = len(test_dataset)
    if num_samples ==0 :
        raise RuntimeError('Test dataset is empty...!')
    
    # Loss function
    criterion = nn.MSELoss(reduction='mean').to(device)
    loss_meter = AverageMeter()
    mace_meter = AverageMeter() #mean average corner error
    elapsed = 0

    
    torch.backends.cudnn.benchmark = pin    #gpu면 cudnn의 가장 빠른 길찾기 모드 ON

    with torch.inference_mode():
        for (img, target) in tqdm(test_loader):
            if pin:
                img = img.to(device, non_blocking=True) #이미지를 gpu로 옮겨라, non_blocking=True : 다른 일도 해라
                target = target.to(device, non_blocking=True)
            else:
                img= img.to(device)
                target = target.to(device)
            
            target = target.float().view(img.size(0), 8)    #정답을 실수형으로 변환, 차원 정리 : 배치 크기 N이고, 각 샘플은 8개숫자로 맞춤

            start = time.time()
            out = model(img)  # [N, 8]
            end = time.time()
            elapsed = elapsed + (end - start)

            out = out.view(out.size(0), -1) #out을 일렬로 정리

            pred_offsets = out * SCALE_PRED
            gt_offsets = target

            #MSE
            loss = criterion(pred_offsets, gt_offsets)
            loss_meter.update(loss.item(), img.size(0))

            #MACE
            delta = (pred_offsets - gt_offsets).view(-1, 4, 2)  #-1 : 배치크기
            ace_batch = delta.norm(dim=2).mean(dim=1)   #코너마다 코너별거리(픽셀) 구하기 ->.mean(dim=1) : 4 코너 거리 평균
            mace_meter.update(ace_batch.mean().item(), img.size(0)) # ACE_BATCH 평균 * 배치 크기


    print('MSE Loss: {:.4f}'.format(loss_meter.avg))
    print('MACE : {:.4f} px'.format(mace_meter.avg))














    # # Batches
    # for (img, target) in tqdm(test_loader):
    #     # Move to CPU, if available
    #     # img = F.interpolate(img, size=(img.size(2) // 2, img.size(3) // 2), mode='bicubic', align_corners=False)
    #     img = img.to(device)  # [N, 3, 128, 128]
    #     target = target.float().to(device)  # [N, 8]

    #     # Forward prop.
    #     with torch.no_grad():
    #         start = time.time()
    #         out = model(img)  # [N, 8]
    #         end = time.time()
    #         elapsed = elapsed + (end - start)

    #     # Calculate loss
    #     out = out.squeeze(dim=1)
    #     loss = criterion(out * 2, target)

    #     losses.update(loss.item(), img.size(0))

    # print('Elapsed: {0:.5f} ms'.format(elapsed / num_samples * 1000))
    # print('Loss: {0:.2f}'.format(losses.avg))