import torch
import torchvision
from thop import profile

from models.model_stages import BiSeNet

backbone = 'STDCNet1446'
use_boundary_2 = False
use_boundary_4 = False
use_boundary_8 = False
use_boundary_16 = False
use_conv_last = False
n_classes = 6

model = BiSeNet(backbone=backbone, n_classes=n_classes,
         use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4,
         use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16,
         use_conv_last=use_conv_last)

dummy_input = torch.randn(1, 1, 448, 704)
flops, params = profile(model, (dummy_input,))
print('FLOPs: ', flops, 'params: ', params)


