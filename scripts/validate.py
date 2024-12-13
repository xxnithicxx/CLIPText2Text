import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def validate(model, loader):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(batch)
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits)
    all_logits = all_logits.numpy()
    all_logits = all_logits.flatten()
    all_logits = (all_logits > 0).astype(int)
    return accuracy_score(loader.dataset.labels, all_logits)