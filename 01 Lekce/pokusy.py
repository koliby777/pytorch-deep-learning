import torch                                        # root package
from torch.utils.data import Dataset, DataLoader    # dataset representation and loading


torch.cuda.is_available                                     # check for cuda
x = x.cuda()                                                # move x's data from
                                                            # CPU to GPU and return new object

x = x.cpu()                                                 # move x's data from GPU to CPU
                                                            # and return new object

if not args.disable_cuda and torch.cuda.is_available():     # device agnostic code
    args.device = torch.device('cuda')                      # and modularity
else:                                                       #
    args.device = torch.device('cpu')                       #

net.to(device)                                              # recursively convert their
                                                            # parameters and buffers to
                                                            # device specific tensors

x = x.to(device)                                            # copy your tensors to a device
                                                            # (gpu, cpu)