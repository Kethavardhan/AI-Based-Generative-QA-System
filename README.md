# Capstone Project - AIML - IIIT HYD

## Team - 19

This repository contains two projects as part of the capstone initiative for artificial intelligence and machine learning. Both projects are designed to demonstrate advanced use cases in natural language processing (NLP), including generating email subject lines and answering AI-related questions. The projects are built using state-of-the-art transformer models, fine-tuned on specific datasets, and deployed using Gradio for interactive user interfaces.

## Table of Contents
- [Email Subject Line Generator](#email-subject-line-generator)
- [AI Q&A Bot](#ai-qna-bot)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Next Steps](#next-steps)

## Email Subject Line Generator

### Project Overview

The **Email Subject Line Generator** project uses the `facebook/bart-large-cnn` model to generate email subject lines from email body texts. This model is a large transformer-based language model that is particularly well-suited for summarization and text generation tasks. We fine-tuned the model to predict concise and relevant subject lines based on a corpus of emails.

- **Model**: `facebook/bart-large-cnn`
- **Task**: Text summarization and generation
- **Deployment**: Gradio for web-based UI
- **Training Platform**: Google Colab (T5 GPU)

### Dataset Description

The dataset for this project contains thousands of email body texts paired with their corresponding subject lines. This dataset is used to fine-tune the BART model, allowing it to learn how to map lengthy email content into short, engaging subject lines. 

- **Dataset**: 
  - Email body text (input)
  - Corresponding subject line (target)

The dataset was preprocessed to remove any special characters, unnecessary whitespaces, and ensure that the length of the email bodies is appropriate for the model's input constraints.

### Metrics

After training the model, we evaluated its performance using common NLP metrics such as ROUGE and BERTScore. These metrics help us assess the overlap between the generated subject lines and the reference subject lines from the dataset.

- **ROUGE-1**: Measures unigram overlap.
- **ROUGE-2**: Measures bigram overlap.
- **ROUGE-L**: Longest common subsequence.
- **METEOR**: Considers precision, recall, and synonym matching.
- **BERTScore F1**: Assesses the semantic similarity of generated and reference text.

### Next Steps

- **Dataset Expansion**: Add more email data across diverse industries to improve subject line variety.
- **Model Exploration**: Fine-tune larger transformer models (e.g., GPT-3) to compare performance.
- **UI Enhancements**: Add additional features to the Gradio interface, such as multi-sentence input support and email templates.

## AI Q&A Bot

### Project Overview

The **AI Q&A Bot** project uses the `GPT-2` model, fine-tuned to answer questions related to artificial intelligence. This project demonstrates the model's ability to generate coherent and relevant answers based on a set of AI-related Q&A pairs. The fine-tuning process allows the model to specialize in answering queries specific to this domain.

- **Model**: `GPT-2` (fine-tuned for Q&A)
- **Task**: Question answering (Q&A)
- **Deployment**: Gradio for web-based UI
- **Training Platform**: Google Colab (T5 GPU)

### Dataset Description

The dataset consists of question and answer pairs related to artificial intelligence. This includes questions about machine learning algorithms, neural networks, deep learning, and various AI techniques. The model is trained to provide accurate, relevant, and contextual answers for these AI-related questions.

- **Dataset**: 
  - Questions related to artificial intelligence (input)
  - Corresponding answers (target)

The dataset was curated to ensure diversity in the types of questions asked and their difficulty levels, ranging from basic concepts to advanced AI topics.

### Metrics

The model's performance was evaluated using standard text generation metrics, providing insights into the quality of the answers it generates.

- **ROUGE-1**: Measures unigram overlap.
- **ROUGE-2**: Measures bigram overlap.
- **ROUGE-L**: Longest common subsequence.
- **METEOR**: Considers precision, recall, and synonym matching.
- **BERTScore F1**: Assesses the semantic similarity of generated and reference text.

These scores reflect how well the model’s answers overlap with the reference answers, as well as their semantic similarity.

### Next Steps

- **Dataset Expansion**: Include questions about broader AI topics or incorporate recent advancements.
- **Model Improvements**: Fine-tune other language models like `facebook/bart-large` or `T5` to compare their performance in Q&A generation.
- **Contextual Enhancements**: Implement multi-turn conversations to allow users to ask follow-up questions and receive more detailed answers.

## Requirements

Both projects use similar libraries and frameworks. To run them, ensure you have the following installed:

- **Google Colab account**: Required for training and deployment.
- **Python Libraries**:
  - `transformers`
  - `torch`
  - `gradio`

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kethavardhan/AI-Based-Generative-QA-System
   cd AI-Based-Generative-QA-System
