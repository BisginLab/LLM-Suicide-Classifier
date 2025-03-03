import torch
import numpy

print(torch.cuda.is_available())  # True if CUDA is available
print(torch.cuda.device_count())  # Number of GPUs
print(torch.cuda.get_device_name(0))  # Name of the first GPU


print(f"CPU Thread Count: {torch.get_num_threads()}")
#Result: 12 threads

