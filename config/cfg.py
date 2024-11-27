import torch

class CFG:
    # Paths
    data_path = "data/questions.csv"
    output_model_path = "outputs/models/best.pt"

    # Hyperparameters
    batch_size = 32
    num_workers = 4
    learning_rate = 1e-3
    weight_decay = 1e-3
    patience = 2
    factor = 0.5
    epochs = 10
    max_length = 128
    projection_dim = 256
    dropout = 0.1
    temperature = 0.07
    text_embedding= 768

    # Model settings
    text_encoder_model = "distilbert-base-uncased"
    text_tokenizer = "distilbert-base-uncased"
    pretrained = True
    trainable = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
