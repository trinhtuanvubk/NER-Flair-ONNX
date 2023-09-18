
import torch
from flair.data import Sentence
from flair.models import SequenceTagger


from flair.embeddings import TransformerWordEmbeddings


def convert_sequence_tagger(model_path):
    model = SequenceTagger.load(model_path)
    example_sentence = Sentence("This is a sentence.")
    longer_sentence = Sentence("This is a way longer sentence to ensure varying lengths work with LSTM.")

    reordered_sentences = sorted([example_sentence, longer_sentence], key=len, reverse=True)
    # rnn paded need order sentences
    tensors = model._prepare_tensors(reordered_sentences)
    # 
    print(tensors[0].shape)
    print(tensors[1].shape)

    print(len(tensors))
    torch.onnx.export(
        model,
        tensors,
        "dyn_sequencetagger2.onnx",
        input_names=["sentence_tensor", "lengths.1"],
        output_names=["scores", "lengths", "319"],
        dynamic_axes={"sentence_tensor" : {0: 'batch_size',
                                           1: 'max_length'},
                    "lengths.1" : {0: 'batch_size'},
                    "scores" : {0: 'batch_size',
                                1: 'max_length',
                                2: 'num_tags',
                                3: 'num_tags'},
                    "lengths" : {0: 'batch_size'},
                    "319": {0: 'num_tags',
                            1: 'num_tags'}},
        opset_version=9,
        verbose=True,
    )




if __name__=="__main__":
    model_path = "./checkpoints/best-model.pt"

    convert_sequence_tagger(model_path)
