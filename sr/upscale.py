import os, cv2, torch
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
torch.set_num_threads(os.cpu_count() or 4)
base = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/'
os.makedirs('sr/out', exist_ok=True)
img = cv2.imread('sr/in/cohesiva.png', cv2.IMREAD_COLOR)
for s in (0.2, 0.4, 0.6):
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    up = RealESRGANer(scale=4, model_path=[base + 'realesr-general-x4v3.pth', base + 'realesr-general-wdn-x4v3.pth'],
                      dni_weight=[s, 1 - s], model=model, tile=0, half=False)
    out, _ = up.enhance(img, outscale=4)
    cv2.imwrite(f'sr/out/cohesiva-general-{int(s*10)}.png', out)
    print(s, out.shape, flush=True)
