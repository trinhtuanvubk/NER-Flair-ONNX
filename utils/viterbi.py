from typing import Tuple

import numpy as np
import torch
import torch.nn
from torch.nn.functional import softmax


import flair
from flair.data import Dictionary, Label, List, Sentence

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


class ViterbiDecoder:
    """
    Decodes a given sequence using the Viterbi algorithm.
    """

    def __init__(self, tag_dictionary: Dictionary):
        """
        :param tag_dictionary: Dictionary of tags for sequence labeling task
        """
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)
        self.start_tag = tag_dictionary.get_idx_for_item(START_TAG)
        self.stop_tag = tag_dictionary.get_idx_for_item(STOP_TAG)

    def decode(
        self, features_tuple: tuple, probabilities_for_all_classes: bool, sentences: List[Sentence]
    ) -> Tuple[List, List]:
        """
        Decoding function returning the most likely sequence of tags.
        :param features_tuple: CRF scores from forward method in shape (batch size, seq len, tagset size, tagset size),
            lengths of sentence in batch, transitions of CRF
        :param probabilities_for_all_classes: whether to return probabilities for all tags
        :return: decoded sequences
        """

        features, lengths, transitions = features_tuple
        features = torch.from_numpy(features).to(flair.device)
        lengths = torch.from_numpy(lengths).to(flair.device)
        transitions = torch.from_numpy(transitions).to(flair.device)
        print(features.shape, lengths.shape, transitions.shape)

        all_tags = []

        
        batch_size = features.size(0)
        seq_len = features.size(1)

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, seq_len + 1, self.tagset_size).to(flair.device)
        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current tag
        # Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = (
            torch.ones((batch_size, seq_len + 1, self.tagset_size), dtype=torch.long, device=flair.device)
            * self.stop_tag
        )

        for t in range(seq_len):
            batch_size_t = sum([length > t for length in lengths])  # effective batch size (sans pads) at this timestep
            terminates = [i for i, length in enumerate(lengths) if length == t + 1]

            if t == 0:
                scores_upto_t[:batch_size_t, t] = features[:batch_size_t, t, :, self.start_tag]
                backpointers[:batch_size_t, t, :] = (
                    torch.ones((batch_size_t, self.tagset_size), dtype=torch.long) * self.start_tag
                )
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and
                # choose the previous timestep that corresponds to the max. accumulated score for each current timestep
                scores_upto_t[:batch_size_t, t], backpointers[:batch_size_t, t, :] = torch.max(
                    features[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t, t - 1].unsqueeze(1), dim=2
                )

            # If sentence is over, add transition to STOP-tag
            if terminates:
                scores_upto_t[terminates, t + 1], backpointers[terminates, t + 1, :] = torch.max(
                    scores_upto_t[terminates, t].unsqueeze(1) + transitions[self.stop_tag].unsqueeze(0), dim=2
                )

        # Decode/trace best path backwards
        decoded = torch.zeros((batch_size, backpointers.size(1)), dtype=torch.long, device=flair.device)
        pointer = torch.ones((batch_size, 1), dtype=torch.long, device=flair.device) * self.stop_tag

        for t in list(reversed(range(backpointers.size(1)))):
            decoded[:, t] = torch.gather(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)

        # Sanity check
        assert torch.equal(
            decoded[:, 0], torch.ones((batch_size), dtype=torch.long, device=flair.device) * self.start_tag
        )

        # remove start-tag and backscore to stop-tag
        scores_upto_t = scores_upto_t[:, :-1, :]
        decoded = decoded[:, 1:]

        # Max + Softmax to get confidence score for predicted label and append label to each token
        scores = softmax(scores_upto_t, dim=2)
        confidences = torch.max(scores, dim=2)

        tags = []
        for tag_seq, tag_seq_conf, length_seq in zip(decoded, confidences.values, lengths):
            tags.append(
                [
                    (self.tag_dictionary.get_item_for_index(tag), conf.item())
                    for tag, conf in list(zip(tag_seq, tag_seq_conf))[:length_seq]
                ]
            )

        if probabilities_for_all_classes:
            all_tags = self._all_scores_for_token(scores.cpu(), lengths, sentences)

        return tags, all_tags

    def _all_scores_for_token(self, scores: torch.Tensor, lengths: torch.IntTensor, sentences: List[Sentence]):
        """
        Returns all scores for each tag in tag dictionary.
        :param scores: Scores for current sentence.
        """
        scores = scores.numpy()
        prob_tags_per_sentence = []
        for scores_sentence, length, sentence in zip(scores, lengths, sentences):
            scores_sentence = scores_sentence[:length]
            prob_tags_per_sentence.append(
                [
                    [
                        Label(token, self.tag_dictionary.get_item_for_index(score_id), score)
                        for score_id, score in enumerate(score_dist)
                    ]
                    for score_dist, token in zip(scores_sentence, sentence)
                ]
            )
        return prob_tags_per_sentence

