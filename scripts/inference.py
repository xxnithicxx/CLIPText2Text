import torch
from transformers import DistilBertTokenizer
from models.clip_model import CLIPModel
from config.cfg import CFG


def infer_similarity(question1, question2, model_path="outputs/models/best.pt"):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    encoded_question1 = tokenizer(
        question1,
        padding="max_length",
        truncation=True,
        max_length=CFG.max_length,
        return_tensors="pt",
    )
    encoded_question2 = tokenizer(
        question2,
        padding="max_length",
        truncation=True,
        max_length=CFG.max_length,
        return_tensors="pt",
    )

    batch = {
        "input_ids_1": encoded_question1["input_ids"].to(CFG.device),
        "attention_mask_1": encoded_question1["attention_mask"].to(CFG.device),
        "input_ids_2": encoded_question2["input_ids"].to(CFG.device),
        "attention_mask_2": encoded_question2["attention_mask"].to(CFG.device),
    }

    with torch.no_grad():
        loss = model(batch)
        similarity_score = torch.exp(-loss).item()

    return similarity_score
