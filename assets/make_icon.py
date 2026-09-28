"""Draws the kSight icon (half k-space, half the brain it becomes, plus the name) from the real scan."""
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mri.pipeline import reconstruct
from mri.raw_data import find_datasets, load_raw_kspace

RUST, DARK, CREAM, GOLD = (154, 70, 50), (96, 40, 28), (248, 246, 242), (242, 165, 65)
S = 1024


def disc(size, kspace, brain):
    """A circle: k-space on the left, the brain on the right, a cream ring and divider."""
    k8 = (colormaps["magma"](kspace)[..., :3]*255).astype(np.uint8)
    left = Image.fromarray(k8).resize((size, size), Image.LANCZOS)
    right = Image.fromarray((brain*255).astype(np.uint8)).convert("RGB").resize((size, size), Image.LANCZOS)
    half = Image.new("L", (size, size), 0); ImageDraw.Draw(half).rectangle([size//2, 0, size, size], fill=255)
    face = Image.composite(right, left, half)
    mask = Image.new("L", (size, size), 0); ImageDraw.Draw(mask).ellipse([0, 0, size - 1, size - 1], fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0)); out.paste(face, (0, 0), mask)
    d = ImageDraw.Draw(out)
    d.line([size//2, 6, size//2, size - 6], fill=CREAM, width=max(2, size//90))
    d.ellipse([0, 0, size - 1, size - 1], outline=CREAM, width=max(2, size//40))
    return out


def tile(size):
    """Rounded square with a rust gradient."""
    grad = np.linspace(0, 1, size)[:, None, None]
    rgb = (np.array(RUST)*(1 - grad) + np.array(DARK)*grad).astype(np.uint8)*np.ones((1, size, 1), np.uint8)
    img = Image.fromarray(rgb).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=size//5, fill=255)
    img.putalpha(mask)
    return img


path = next(p for p in find_datasets() if p.name == "2022062501_T201.h5")
o = reconstruct(None, {}, kspace=load_raw_kspace(str(path), 9))
ks = o["kspace"][:, 30:226]                              # the measured columns only
ks = np.asarray(Image.fromarray((ks*255).astype(np.uint8)).resize((256, 256), Image.LANCZOS))/255
lo = (np.median(ks) - 0.3)/0.7                            # median (the noise) lands on dim purple
ks = np.clip((ks - lo)/(1 - lo), 0, 1)
brain = o["recon"][14:242, 14:242]                       # the whole head inside the circle
brain = np.clip((brain - brain.min())/(np.percentile(brain, 99.7) - brain.min()), 0, 1)

# full icon: disc + name
icon = tile(S)
d_ = disc(620, ks, brain)
shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ImageDraw.Draw(shadow).ellipse([202, 112, 202 + 620, 112 + 620], fill=(0, 0, 0, 110))
icon = Image.alpha_composite(icon, shadow.filter(ImageFilter.GaussianBlur(18)))
icon.alpha_composite(d_, (202, 92))
draw = ImageDraw.Draw(icon)
font = ImageFont.truetype("C:/Windows/Fonts/GOTHICB.TTF", 190)
k_w = draw.textlength("k", font=font); rest_w = draw.textlength("Sight", font=font)
x0, y0 = (S - k_w - rest_w)/2, 752
draw.text((x0, y0), "k", font=font, fill=GOLD)
draw.text((x0 + k_w, y0), "Sight", font=font, fill=CREAM)
icon.save(HERE/"icon.png")

# small sizes: the disc alone reads better than tiny letters
mark = tile(S); mark.alpha_composite(disc(800, ks, brain), (112, 112))
sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
mark.save(HERE/"icon.ico", sizes=sizes[:4])
icon.save(HERE/"icon_large.ico", sizes=sizes[4:])
mark.save(HERE/"mark.png")
print("saved", [p.name for p in HERE.glob("*.png")] + [p.name for p in HERE.glob("*.ico")])
