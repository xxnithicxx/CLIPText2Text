import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import torch
import itertools
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from models.clip_model import CLIPModel, SigLIPModel
from datasets.clip_dataset import CLIPTextDataset
from transformers import DistilBertTokenizer
from config.cfg import CFG
from validate import validate

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    loss_meter = 0
    for batch in tqdm(loader):
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
    scheduler.step(loss_meter / len(loader))
    return loss_meter / len(loader)


def main():
    df = pd.read_csv("data/questions.csv")
    train_df = df.sample(frac=0.8, random_state=42)
    valid_df = df.drop(train_df.index)

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

    train_dataset = CLIPTextDataset(train_df, tokenizer=tokenizer)
    valid_dataset = CLIPTextDataset(valid_df, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size)

    # model = CLIPModel().to(CFG.device)
    model = SigLIPModel().to(CFG.device)
    optimizer = AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=CFG.patience, factor=CFG.factor)

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch + 1}/{CFG.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        valid_loss = validate(model, valid_loader)

        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "outputs/models/best.pt")
            print("Saved Best Model!")


if __name__ == "__main__":
    main()
