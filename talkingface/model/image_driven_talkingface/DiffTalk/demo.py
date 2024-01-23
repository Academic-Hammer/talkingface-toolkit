from talkingface.model.image_driven_talkingface.DiffTalk import get_difftalk_inference
import os
from talkingface.model.image_driven_talkingface.DiffTalk.ldm.models.diffusion.ddim_ldm_ref_inpaint import DDIMSampler
import torch
from einops import rearrange
from PIL import Image
import numpy as np

model = get_difftalk_inference()

save_dir = "./video/"
ddim_steps = 50
batch = None  # load your data here
batchsize = 1
ddim_eta = 0  # Langevin
print(model)
sampler = DDIMSampler(model)
samples = []
samples_inpainting = []
xrec_img = []
z, c_audio, c_lip, c_ldm, c_mask, x, xrec, xc_audio, xc_lip = model.get_input(batch, 'image',
                                                                              return_first_stage_outputs=True,
                                                                              force_c_encode=True,
                                                                              return_original_cond=True,
                                                                              bs=batchsize)
shape = (z.shape[1], z.shape[2], z.shape[3])
N = x.shape[0]
c = {'audio': c_audio, 'lip': c_lip, 'ldm': c_ldm, 'mask_image': c_mask}

b, h, w = z.shape[0], z.shape[2], z.shape[3]
landmarks = batch["landmarks_all"]
landmarks = landmarks / 4
mask = batch["inference_mask"].to(model.device)
mask = mask[:, None, ...]
with model.ema_scope():
    samples_ddim, _ = sampler.sample(ddim_steps, N, shape, c, x0=z[:N], verbose=False, eta=ddim_eta, mask=mask)

x_samples_ddim = model.decode_first_stage(samples_ddim)
x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)
samples_inpainting.append(x_samples_ddim)

# save images
samples_inpainting = torch.stack(samples_inpainting, 0)
samples_inpainting = rearrange(samples_inpainting, 'n b c h w -> (n b) c h w')
save_path = os.path.join(save_dir, '105_a105_mask_face')
if not os.path.exists(save_path):
    os.mkdir(save_path)
for j in range(samples_inpainting.shape[0]):
    samples_inpainting_img = 255. * rearrange(samples_inpainting[j], 'c h w -> h w c').cpu().numpy()
    img = Image.fromarray(samples_inpainting_img.astype(np.uint8))
    img.save(os.path.join(save_path, '{:04d}_{:04d}.jpg'.format(0, j)))