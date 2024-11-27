import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from config.cfg import CFG


class CLIPTextDataset(Dataset):
    """
    Dataset class for text-to-text similarity data.
    This processes questions and their duplication labels.
    """

    def __init__(self, dataframe, tokenizer=None, max_length=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
            tokenizer (transformers.Tokenizer): Tokenizer to encode questions.
            max_length (int): Maximum length of tokenized sequences.
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer or DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
        self.max_length = max_length or CFG.max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Encode question1
        encoded_question1 = self.tokenizer(
            row["question1"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Encode question2
        encoded_question2 = self.tokenizer(
            row["question2"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create a dictionary to hold the data
        item = {
            "input_ids_1": encoded_question1["input_ids"].squeeze(0),  # Shape: (max_length,)
            "attention_mask_1": encoded_question1["attention_mask"].squeeze(0),  # Shape: (max_length,)
            "input_ids_2": encoded_question2["input_ids"].squeeze(0),  # Shape: (max_length,)
            "attention_mask_2": encoded_question2["attention_mask"].squeeze(0),  # Shape: (max_length,)
            "label": torch.tensor(row["is_duplicate"], dtype=torch.float),  # Shape: Scalar
        }

        return item
