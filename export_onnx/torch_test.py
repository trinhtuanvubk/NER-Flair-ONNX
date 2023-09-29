from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings


# model_path = "./checkpoints/best-model.pt"

def torch_test(args)
    model = SequenceTagger.load(args.model_path)

    assert isinstance(model.embeddings, TransformerWordEmbeddings)

    sentences = [Sentence("to speak to a customer service advisor"), Sentence("to speak to a customer")]

    model.predict(sentences)

    for sentence in sentences:
        for entity in sentence.get_spans('ner'):
            print(f"Text: {sentence}")
            print(f"Entity: {entity} - {entity.tag}")