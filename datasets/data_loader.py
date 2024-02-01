import random
import torch
import numpy as np

def cycle_dataloader(argus, dataset, train_batch_size):
    random.seed(argus.seed)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True)
    while True:
        for data in dataset_loader:
            yield data
        print("Finish this epoch dataloader !!!!!!!")
        random.seed(np.random.randint(0, 9999))
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True)
        random.seed(argus.seed)

