import torch
import torchvision
print(torch.cuda.is_available())     # should be True
print(torchvision.ops.nms)           # should not throw error
print(torch.__version__) 