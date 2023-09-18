
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='NER ONNX')
    parser.add_argument('--scenario', type=str, default='train')
    parser.add_argument('--tag_type', type=str, default='ner')
    parser.add_argument('--tag_format', type=str, default='BIO')
    parser.add_argument('--model_name', type=str, default="bert-base-uncased", help='model name')
    parser.add_argument('--embedding_name', type=str, default="bert-base-uncased", help='embedding name')
    parser.add_argument('--sentence', type=str, default="to check my bill", help='sentence test for inference')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args