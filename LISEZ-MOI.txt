# Vision Classifier

A few-shot vision classifier package that can be used with different encoders like Hugging Face and Ollama.

## Installation

Install the core package with:

```bash
pip install vision-classifier
```

To use the Hugging Face encoder, install the `hf` extra:

```bash
pip install vision-classifier[hf]
```

To use the Ollama encoder, install the `ollama` extra:

```bash
pip install vision-classifier[ollama]
```

## Usage

The main goal of this package is to provide a simple interface for few-shot image classification with no or very little training and data.
You can easily test multiple pre-trained encoders and reserve the real classification-part to known, lightweight algorithms like kNN, SVM or random forests.

The true inspiration for this package is to limite the number of images needed for a classification and limite as much as possible the training time. You can run this classifier on CPU while using all the knowledge of the pre-trained encoders such as CLIP, ViT, etc.
