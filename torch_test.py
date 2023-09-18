import os
import transformers
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings


ckpt = "./checkpoints/best-model.pt"

model = SequenceTagger.load(ckpt)


assert isinstance(model.embeddings, TransformerWordEmbeddings)

sentences = [Sentence("to speak to a customer service advisor"), Sentence("to speak to a customer")]

model.predict(sentences)

for sentence in sentences:
    for entity in sentence.get_spans('ner'):
        print(f"Text: {sentence}")
        print(f"Entity: {entity} - {entity.tag}")