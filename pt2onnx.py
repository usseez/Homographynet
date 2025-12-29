# ts2onnx.py (fixed)
import argparse
import torch
import torch.nn as nn
from mobilenet_v2 import MobileNetV2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", default="./weights/homonet.pt", help="state_dict .pt path")
    parser.add_argument("--output_name", default="model.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--input-ch", type=int, default=3)     # ★ MobileNetV2 기본 3채널
    parser.add_argument("--input-h", type=int, default=128)
    parser.add_argument("--input-w", type=int, default=128)
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic axes")
    parser.add_argument("--input", default="input")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) 모델 생성 & 가중치 로드
    model = MobileNetV2().to(device)
    ckpt = torch.load(args.pt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[load] missing={list(missing)} unexpected={list(unexpected)}")
    model.eval()

    # 2) Quant/DeQuant을 ONNX 친화적으로 대체
    model.quant = nn.Identity()
    model.dequant = nn.Identity()

    # 3) 입력 형태 점검 (첫 Conv가 3채널 기대)
    first_conv = model.features[0][0]  # ConvBNReLU -> [0]=Conv2d
    exp_in_ch = first_conv.in_channels
    if exp_in_ch != args.input_ch:
        print(f"[warn] model expects input_ch={exp_in_ch}, but got --input-ch={args.input_ch}. "
              f"Use --input-ch {exp_in_ch} (권장)")

    # 4) 더미 입력
    dummy = torch.randn(1, args.input_ch, args.input_h, args.input_w, dtype=torch.float32, device=device)
    print(f"[info] dummy.shape={tuple(dummy.shape)}  first_conv.in_ch={exp_in_ch}")

    # 5) dynamic axes
    dynamic_axes = None if args.no_dynamic else {
        args.input: {0: "batch", 2: "height", 3: "width"},
        args.output: {0: "batch"},
    }

    # 6) export
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            args.output_name,
            input_names=[args.input],
            output_names=[args.output],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )
    print(f"[OK] Exported: {args.output_name} (opset={args.opset}, shape={tuple(dummy.shape)})")

if __name__ == "__main__":
    main()
