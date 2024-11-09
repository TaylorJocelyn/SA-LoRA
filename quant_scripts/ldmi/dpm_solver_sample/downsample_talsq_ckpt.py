import torch
ckpt = torch.load('reproduce/ldmi/weight/saqlora/quantw4a4_fporder1_qorder1_100steps_saqlora_177epochs_dpmsolver.pth', map_location='cpu')
newsd = {}
for k, v in ckpt.items():
    if 'delta_list' in k or 'zp_list' in k:
        # newv = v[::12][:20]
        newv = v[::5]
    
    else:
        newv = v
    
    newsd[k] = newv

torch.save(newsd, 'reproduce/ldmi/weight/saqlora/quantw4a4_fporder1_qorder1_100steps_saqlora_177epochs_dpmsolver_sample20.pth')