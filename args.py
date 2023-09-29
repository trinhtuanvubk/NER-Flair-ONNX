
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='NER ONNX')
    parser.add_argument('--scenario', type=str, default='tagger')
    parser.add_argument('--embedding_path', type=str, default='./checkpoints/embedding.onnx')
    parser.add_argument('--tagger_path', type=str, default='./checkpoints/tagger.onnx')
    parser.add_argument('--tag_dictionary_path', type=str, default='./tag_dictionary/tag_dictionary.pkl')
    parser.add_argument('--tag_type', type=str, default='ner')
    parser.add_argument('--tag_format', type=str, default='BIO')
    parser.add_argument('--model_path', type=str, default="./checkpoints/best-model.pt", help='model path')
    parser.add_argument('--embedding_name', type=str, default="bert-base-uncased", help='embedding name')


    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args