"""MONAI reference for tools/prep-test.mjs: saves the processor's crop and prints its lung bounds.
  venv/bin/python tools/prep-ref.py CT.nii.gz OUT.npy [chest|abdomen]"""
import sys, warnings
import numpy as np
sys.path.insert(0, '/data/huggingface/nv-reason-ct/hf')
from processor import ImageLoader3D, AnatomySpatialCropd
path, out, region = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'abdomen'
loader = ImageLoader3D()
orig = AnatomySpatialCropd._detect_lung_bounds
def logged(self, image):
    b = orig(self, image); print('shape', tuple(image.shape), 'bounds', b); return b
AnatomySpatialCropd._detect_lung_bounds = logged
img = loader.load_image(path, anatomy_region=region)
np.save(out, img.numpy().astype(np.float32))
print('crop', tuple(img.shape))
