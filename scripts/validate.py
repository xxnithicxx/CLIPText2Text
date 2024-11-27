import torch
from tqdm import tqdm


def validate(model, loader):
    model.eval()
    loss_meter = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(batch)
            loss_meter += loss.item()
    return loss_meter / len(loader)
