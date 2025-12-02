import torch

print("\n===== GPU CHECK =====")
print("Torch CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Torch Version:", torch.__version__)
