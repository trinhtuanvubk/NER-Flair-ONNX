
import torch
from flair.data import Sentence
from flair.models import SequenceTagger


def convert_sequence_tagger(args):
    model = SequenceTagger.load(args.model_path)
    example_sentence = Sentence("This is a sentence.")
    longer_sentence = Sentence("This is a way longer sentence to ensure varying lengths work with LSTM.")

    # rnn paded need order sentences
    reordered_sentences = sorted([example_sentence, longer_sentence], key=len, reverse=True)

    tensors = model._prepare_tensors(reordered_sentences)
    dynamic_axes = {"sentence_tensor" : {0: 'batch_size',
                                           1: 'max_length'},
                    "lengths_in" : {0: 'batch_size'},
                    "scores" : {0: 'batch_size',
                                1: 'max_length',
                                2: 'num_tags',
                                3: 'num_tags'},
                    "lengths_out" : {0: 'batch_size'},
                    "transitions" : {0: 'num_tags',
                                    1: 'num_tags'}}
    
    torch.onnx.export(
        model,
        tensors,
        args.tagger_path,
        input_names=["sentence_tensor", "lengths_in"],
        output_names=["scores", "lengths_out", "transitions"],
        dynamic_axes=dynamic_axes,
        opset_version=9,
        verbose=True,
    )



