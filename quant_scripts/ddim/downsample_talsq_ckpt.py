import torch
ckpt = torch.load('reproduce/ldm/weight/quantw4a4_100steps_efficientdm.pth', map_location='cpu')
newsd = {}
for k, v in ckpt.items():
    if 'delta_list' in k or 'zp_list' in k:
        # newv = v[::12][:20]
        newv = v[::2]
    
    else:
        newv = v
    
    newsd[k] = newv

torch.save(newsd, 'reproduce/ldm/weight/quantw4a4_100steps_efficientdm_sample50.pth')