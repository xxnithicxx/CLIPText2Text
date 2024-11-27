# Text-to-Text CLIP Model

This project trains a CLIP-like model for text-to-text similarity tasks, using questions to predict duplication.

## Project Structure

```
project_root/
├── config/               # Configuration files
├── data/                 # Data storage
├── models/               # Model components
├── datasets/             # Dataset loader
├── transforms/           # Data augmentation/transformation functions
├── outputs/              # Model checkpoints and logs
├── scripts/              # Training, validation, and inference scripts
├── requirements.txt      # Python dependencies
├── run_training.sh       # Bash script to run training
├── README.md             # Project documentation
└── .gitignore            # Ignored files/folders
```

## Getting Started

### Prerequisites

- Python >= 3.8
- PyTorch
- Transformers (Hugging Face)
- Additional libraries: tqdm, pandas, numpy, etc.

Install dependencies with:
```bash
pip install -r requirements.txt
```

### Data Preparation

Place your CSV file (`questions.csv`) in the `data/` folder. The CSV should have the following structure:
```
id,qid1,qid2,question1,question2,is_duplicate
```

### Training

Run the training script:
```bash
bash run_training.sh
```

### Inference

Use the `inference.py` script to predict similarity between two questions:
```python
from scripts.inference import infer_similarity

question1 = "What is machine learning?"
question2 = "Explain machine learning concepts."
score = infer_similarity(question1, question2)
print(f"Similarity score: {score}")
```

### Results

The best model will be saved in the `outputs/models/` directory.

## License

This project is open-source and available under the MIT License.