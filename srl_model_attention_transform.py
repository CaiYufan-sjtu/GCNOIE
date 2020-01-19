import logging
from typing import Dict, List, Iterable, Tuple, Any, Optional
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence
import warnings

import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
from torch.nn.modules import Linear, Dropout
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
import numpy as np
import scipy.sparse as sp

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.srl_util import convert_bio_tags_to_conll_format, write_bio_formatted_tags_to_file
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics.srl_eval_scorer import SrlEvalScorer, DEFAULT_SRL_EVAL_PATH

from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _convert_tags_to_wordpiece_tags(tags: List[str], offsets: List[int]) -> List[str]:
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.
    This is only used if you pass a `bert_model_name` to the dataset reader below.
    Parameters
    ----------
    tags : `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces
    offsets : `List[int]`
        The wordpiece offsets.
    Returns
    -------
    The new BIO tags.
    """
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        is_o = tag == "O"
        is_start = True
        while j < offset:
            if is_o:
                new_tags.append("O")

            elif tag.startswith("I"):
                new_tags.append(tag)

            elif is_start and tag.startswith("B"):
                new_tags.append(tag)
                is_start = False

            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                new_tags.append("I-" + label)
            j += 1

    # Add O tags for cls and sep tokens.
    return ["O"] + new_tags + ["O"]


def _convert_verb_indices_to_wordpiece_indices(verb_indices: List[int], offsets: List[int]): # pylint: disable=invalid-name
    """
    Converts binary verb indicators to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.
    This is only used if you pass a `bert_model_name` to the dataset reader below.
    Parameters
    ----------
    verb_indices : `List[int]`
        The binary verb indicators, 0 for not a verb, 1 for verb.
    offsets : `List[int]`
        The wordpiece offsets.
    Returns
    -------
    The new verb indices.
    """
    j = 0
    new_verb_indices = []
    for i, offset in enumerate(offsets):
        indicator = verb_indices[i]
        while j < offset:
            new_verb_indices.append(indicator)
            j += 1

    # Add 0 indicators for cls and sep tokens.
    return [0] + new_verb_indices + [0]


class SrlReader(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling. It returns a dataset of instances with the
    following fields:
    tokens : ``TextField``
        The tokens in the sentence.
    verb_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.
    bert_model_name : ``Optional[str]``, (default = None)
        The BERT model to be wrapped. If you specify a bert_model here, then we will
        assume you want to use BERT throughout; we will use the bert tokenizer,
        and will expand your tags and verb indicators accordingly. If not,
        the tokens will be indexed as normal with the token_indexers.
    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False,
                 bert_model_name: str = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier

        if bert_model_name is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name
        else:
            self.bert_tokenizer = None
            self.lowercase_input = False

    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the begining and end of the sentence.
        A slight oddity with this function is that it also returns the wordpiece offsets
        corresponding to the _start_ of words as well as the end.
        We need both of these offsets (or at least, it's easiest to use both), because we need
        to convert the labels to tags using the end_offsets. However, when we are decoding a
        BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
        because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
        wordpieces (this happens in the case that a word is split into multiple word pieces,
        and then we take the last tag of the word, which might correspond to, e.g, I-V, which
        would not be allowed as it is not preceeded by a B tag).
        For example:
        `annotate` will be bert tokenized as ["anno", "##tate"].
        If this is tagged as [B-V, I-V] as it should be, we need to select the
        _first_ wordpiece label to be the label for the token, because otherwise
        we may end up with invalid tag sequences (we cannot start a new tag with an I).
        Returns
        -------
        wordpieces : List[str]
            The BERT wordpieces from the words in the sentence.
        end_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each word being chosen.
        start_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)

        for sentence in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):
            tokens = [Token(t) for t in sentence.words]
            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = ["O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                yield self.text_to_instance(tokens, verb_label, tags)
            else:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    # for i in range(len(tags)):
                    #     if tags[i] != 'O':
                            # tags[i] = 'I-ARG1'
                    yield self.text_to_instance(tokens, verb_indicator, tags)

    @staticmethod
    def _ontonotes_subset(ontonotes_reader: Ontonotes,
                          file_path: str,
                          domain_identifier: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if domain_identifier is None or f"/{domain_identifier}/" in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        metadata_dict: Dict[str, Any] = {}
        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input([t.text for t in tokens])
            new_verbs = _convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
            metadata_dict["offsets"] = start_offsets
            # In order to override the indexing mechanism, we need to set the `text_id`
            # attribute directly. This causes the indexing to use this id.
            text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                                   token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(new_verbs, text_field)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(verb_label, text_field)

        fields: Dict[str, Field] = {}
        fields['tokens'] = text_field
        fields['verb_indicator'] = verb_indicator

        if all([x == 0 for x in verb_label]):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index

        if tags:
            if self.bert_tokenizer is not None:
                new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                fields['tags'] = SequenceLabelField(new_tags, text_field)
            else:
                fields['tags'] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)


class SemanticRoleLabeler(Model):
    """
    This model performs semantic role labeling using BIO tags using Propbank semantic roles.
    Specifically, it is an implementation of `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .
    This implementation is effectively a series of stacked interleaved LSTMs with highway
    connections, applied to embedded sequences of words concatenated with a binary indicator
    containing whether or not a word is the verbal predicate to generate predictions for in
    the sentence. Additionally, during inference, Viterbi decoding is applied to constrain
    the predictions to contain valid BIO sequences.
    Specifically, the model expects and outputs IOB2-formatted tags, where the
    B- tag is used in the beginning of every chunk (i.e. all chunks start with the B- tag).
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    binary_feature_dim : int, required.
        The dimensionality of the embedding of the binary verb predicate features.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    label_smoothing : ``float``, optional (default = 0.0)
        Whether or not to use label smoothing on the labels when computing cross entropy loss.
    ignore_span_metric: ``bool``, optional (default = False)
        Whether to calculate span loss, which is irrelevant when predicting BIO for Open Information Extraction.
    srl_eval_path: ``str``, optional (default=``DEFAULT_SRL_EVAL_PATH``)
        The path to the srl-eval.pl script. By default, will use the srl-eval.pl included with allennlp,
        which is located at allennlp/tools/srl-eval.pl . If ``None``, srl-eval.pl is not used.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False,
                 srl_eval_path: str = DEFAULT_SRL_EVAL_PATH) -> None:
        super(SemanticRoleLabeler, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")

        if srl_eval_path is not None:
            # For the span based evaluation, we don't want to consider labels
            # for verb, because the verb index is provided to the model.
            self.span_metric = SrlEvalScorer(srl_eval_path, ignore_classes=["V"])
            print(self.span_metric)
        else:
            self.span_metric = None

        self.encoder = encoder
        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes))
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric

        check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim,
                               encoder.get_input_dim(),
                               "text embedding dim + verb indicator embedding dim",
                               "encoder input dim")
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence and the verb to compute the
            frame for, under 'words' and 'verb' keys, respectively.
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens))
        mask = get_text_field_mask(tokens)
        embedded_verb_indicator = self.binary_feature_embedding(verb_indicator.long())
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, embedded_verb_indicator], -1)
        batch_size, sequence_length, _ = embedded_text_with_verb_indicator.size()

        encoded_text = self.encoder(embedded_text_with_verb_indicator, mask)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask

        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits,
                                                      tags,
                                                      mask,
                                                      label_smoothing=self._label_smoothing)
            if not self.ignore_span_metric and self.span_metric is not None and not self.training:
                batch_verb_indices = [example_metadata["verb_index"] for example_metadata in metadata]
                batch_sentences = [example_metadata["words"] for example_metadata in metadata]
                # Get the BIO tags from decode()
                # TODO (nfliu): This is kind of a hack, consider splitting out part
                # of decode() to a separate function.
                batch_bio_predicted_tags = self.decode(output_dict).pop("tags")
                batch_conll_predicted_tags = [convert_bio_tags_to_conll_format(tags) for
                                              tags in batch_bio_predicted_tags]
                batch_bio_gold_tags = [example_metadata["gold_tags"] for example_metadata in metadata]
                batch_conll_gold_tags = [convert_bio_tags_to_conll_format(tags) for
                                         tags in batch_bio_gold_tags]
                self.span_metric(batch_verb_indices,
                                 batch_sentences,
                                 batch_conll_predicted_tags,
                                 batch_conll_gold_tags)
            output_dict["loss"] = loss

        if metadata is not None:
            words, verbs = zip(*[(x["words"], x["verb"]) for x in metadata])
            output_dict["words"] = list(words)
            output_dict["verb"] = list(verbs)
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()
        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix,
                                                        allowed_start_transitions=start_transitions)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            # Return an empty dictionary if ignoring the
            # span metric
            return {}

        else:
            metric_dict = self.span_metric.get_metric(reset=reset)

            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            return {x: y for x, y in metric_dict.items() if "overall" in x}

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.
        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    def get_start_transitions(self):
        """
        In the BIO sequence, we cannot start the sequence with an I-XXX tag.
        This transition sequence is passed to viterbi_decode to specify this constraint.
        Returns
        -------
        start_transitions : torch.Tensor
            The pairwise potentials between a START token and
            the first token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I":
                start_transitions[i] = float("-inf")

        return start_transitions


reader = SrlReader()
train_dataset = reader.read(cached_path("data/train"))
validation_dataset = reader.read(cached_path("data/dev"))
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=100,
                            pretrained_file="https://s3-us-west-2.amazonaws.com/allennlp/"
                                            "datasets/glove/glove.6B.100d.txt.gz", trainable=True)
source_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
# encoder = PytorchSeq2SeqWrapper(StackedAlternatingLstm(input_size=200,
#                                                        hidden_size=300,
#                                                        num_layers=4,
#                                                        recurrent_dropout_probability=0.1,
#                                                        use_highway=True))
encoder = StackedSelfAttentionEncoder(
    input_dim=200,
    hidden_dim=300,
    projection_dim=128,
    feedforward_hidden_dim=128,
    num_layers=1,
    num_attention_heads=8)

# attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
# attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
attention = DotProductAttention()
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
max_decoding_steps = 50   # TODO: make this variable
ZH_EMBEDDING_DIM = 10
model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=ZH_EMBEDDING_DIM,
                          target_namespace='labels',
                          attention=attention,
                          beam_size=8,
                          use_bleu=True)

# model = SemanticRoleLabeler(vocab, source_embedder, encoder, binary_feature_dim=100)
optimizer = optim.Adadelta(model.parameters(), rho=0.95)
iterator = BucketIterator(batch_size=100, sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(
                model=model,
                optimizer=optimizer,
                iterator=iterator,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                validation_metric="+f1-measure-overall",
                grad_clipping=1.0,
                patience=50,
                num_epochs=500,
                cuda_device=-1)
trainer.train()