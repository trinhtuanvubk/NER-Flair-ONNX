from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import inspect
import torch

def convert_embedding(args):

    model = SequenceTagger.load(args.model_path).to(torch.device('cuda'))
    embedding = model.embeddings
    assert isinstance(embedding, (TransformerWordEmbeddings))

    sentences = [Sentence("to speak to a customer service advisor"), Sentence("to speak to a customer")]


    # model.embeddings = model.embeddings.export_onnx("flert-embeddings_2.onnx", sentences, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    example_tensors = embedding.prepare_tensors(sentences)
    print(example_tensors)
    dynamic_axes = {"input_ids": {0: 'batch', 1: "seq_length"},
                    "token_lengths": {0: 'sent-count'},
                    "attention_mask": {0: "batch", 1: "seq_length"},
                    "overflow_to_sample_mapping": {0: "batch"},
                    "word_ids": {0: "batch", 1: "seq_length"},
                    "token_embeddings": {0: "sent-count", 1: "max_token_count", 2: "token_embedding_size"}}
    
    output_names = ["token_embeddings"]


    desired_keys_order = [
        param for param in inspect.signature(embedding.forward).parameters.keys() if param in example_tensors
    ]
    torch.onnx.export(
        embedding,
        (example_tensors,),
        "./checkpoints/embedding.onnx",
        input_names=desired_keys_order,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )