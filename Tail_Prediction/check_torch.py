import sys
try:
    import torch
    print('torch', torch.__version__)
    print('cuda available', torch.cuda.is_available())
except Exception as e:
    print('torch import failed:', e)
print('python', sys.version)
