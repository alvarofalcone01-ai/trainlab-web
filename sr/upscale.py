import glob, os, cv2, numpy as np, torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
torch.set_num_threads(os.cpu_count() or 4)
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
up = RealESRGANer(scale=4, model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                  model=model, tile=256, tile_pad=12, pre_pad=0, half=False)
jobs = [('sr/in/cohesiva.png', 'cohesiva', 1)]  # (path, name, prescale)
for p in sorted(glob.glob('assets/d3/*.webp')) + ['assets/image-2.png', 'assets/image-3.png'] + sorted(glob.glob('media/*.png')):
    jobs.append((p, os.path.splitext(os.path.basename(p))[0], 0.5))
os.makedirs('sr/out', exist_ok=True)
for path, name, pre in jobs:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print('skip', path); continue
    if pre != 1:
        img = cv2.resize(img, None, fx=pre, fy=pre, interpolation=cv2.INTER_AREA)
    out, _ = up.enhance(img, outscale=4)
    h, w = out.shape[:2]; s = min(1.0, 1800 / max(h, w))
    if s < 1: out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'sr/out/{name}.png', out)
    print(name, img.shape, '->', out.shape, flush=True)
