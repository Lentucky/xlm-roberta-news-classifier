# Text Classification Model - Prediction Script

A fine-tuned transformer-based text classification model for Filipino news articles, with an interactive command-line interface for real-time predictions.

## Overview

This script provides an easy-to-use interface for classifying text using a pre-trained transformer model. It loads the model weights, tokenizer, and label mappings to perform inference on any user-provided text input.

## Features

- 🚀 **Real-time Classification**: Interactive terminal interface for instant predictions
- 📊 **Top-K Predictions**: Displays multiple probable categories with confidence scores
- 🔧 **Easy to Use**: Simple command-line interface requiring no ML expertise
- 🎯 **Fine-tuned Model**: Optimized for Filipino news article classification

## Dataset

This model was trained on the **BalitaNLP Dataset**, a comprehensive collection of Filipino news articles.

📁 Dataset Repository: [BalitaNLP-Dataset](https://github.com/KenrickLance/BalitaNLP-Dataset)

## Training

For details on how this model was trained, including hyperparameters, data preprocessing, and evaluation metrics, visit:

📓 Training Notebook: [News Classifier using Balita NLP](https://www.kaggle.com/code/lentucky/news-classifier-using-balita-nlp)

## Requirements

```bash
pip install torch transformers
```

This doesnt contain the best_model.pt and final_model itself, to get it download it here:
https://www.kaggle.com/code/lentucky/news-classifier-using-balita-nlp/output?scriptVersionId=269250257

## Directory Structure

```
.
├── predict.py           # Main prediction script
├── final_model/         # Model directory
│   ├── config.json      # Model configuration
│   ├── pytorch_model.bin # Model weights
│   ├── tokenizer_config.json
│   ├── vocab.txt        # Tokenizer vocabulary
│   └── label_map.json   # Category labels mapping
└── README.md
```

## Usage

### Basic Usage

Run the script and enter text when prompted:

```bash
python predict.py
```

### Example Session

```
$ python predict.py
Loading model and tokenizer...
Loading label map...
Model ready. Type a sentence to classify (or type 'exit' to quit).

Enter text: Ang presyo ng bigas ay tumaas ngayong linggo.

Top Predictions:
business: 0.8521
economy: 0.0892
politics: 0.0345
lifestyle: 0.0156
sports: 0.0086

Enter text: exit
```

### Customizing Top-K Predictions

By default, the script shows the top 5 predictions. You can modify this in the code:

```python
predict_text(text, top_k=3)  # Show only top 3 predictions
```

## How It Works

1. **Model Loading**: Loads the fine-tuned transformer model and tokenizer from the `final_model` directory
2. **Label Mapping**: Reads the `label_map.json` file to map numeric predictions to category names
3. **Text Processing**: Tokenizes input text with proper padding and truncation
4. **Inference**: Passes tokens through the model to generate logits
5. **Probability Calculation**: Applies softmax to convert logits into confidence scores
6. **Results Display**: Shows top-k categories with their respective probabilities

## Function Reference

### `predict_text(text, top_k=5)`

Classifies input text and displays top predictions.

**Parameters:**
- `text` (str): Input sentence or paragraph to classify
- `top_k` (int, optional): Number of top predictions to display. Default: 5

**Returns:**
- None (prints results to console)

## Model Details

- **Architecture**: Transformer-based sequence classification
- **Framework**: PyTorch + Hugging Face Transformers
- **Language**: Filipino (Tagalog)
- **Task**: Multi-class text classification

## Notes

- The model expects Filipino text input for optimal performance
- Long texts are automatically truncated to the model's maximum length
- The model runs in evaluation mode (no training/gradient computation)
- All predictions include confidence scores between 0 and 1

## License

This model can be completely used for free, but still consider the dataset and the model's licenses.

## Citation

If you use this model or the BalitaNLP dataset in your research, please cite:

```
BalitaNLP Dataset: https://github.com/KenrickLance/BalitaNLP-Dataset
```

## Support

For issues related to:
- **Dataset**: Visit the [BalitaNLP repository](https://github.com/KenrickLance/BalitaNLP-Dataset)
- **Training Process**: Check the [Kaggle notebook](https://www.kaggle.com/code/lentucky/news-classifier-using-balita-nlp)
- **Prediction Script**: Open an issue in your repository

---

