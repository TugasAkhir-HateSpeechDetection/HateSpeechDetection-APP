import torch
state_dict = torch.load("app/models/Bi-GRU.pt", map_location='cpu')
for k, v in state_dict.items():
    print(k, v.shape)