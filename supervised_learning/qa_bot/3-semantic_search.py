#!/usr/bin/env python3
"""
Semantic Search module
"""
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def mean_pooling(model_output, attention_mask):
    """Performs mean pooling on token embeddings using attention mask."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents to find the most
    similar document to the given sentence.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2'
    )
    model = AutoModel.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2'
    )

    documents = []
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md') or filename.endswith('.txt'):
            file_path = os.path.join(corpus_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    if not documents:
        return None

    encoded_input = tokenizer(
        [sentence] + documents,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(
        model_output, encoded_input['attention_mask']
    )
    embeddings = F.normalize(embeddings, p=2, dim=1)

    query_embedding = embeddings[0:1]
    doc_embeddings = embeddings[1:]

    scores = torch.mm(query_embedding, doc_embeddings.transpose(0, 1))[0]
    best_idx = torch.argmax(scores).item()

    return documents[best_idx]
