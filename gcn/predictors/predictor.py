from typing import List

from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token

from gcn.dataset_readers.reader import GCN_reader


@Predictor.register('gcn_predictor')
class GCN_Predictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.dependency_tree_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/"
                                                             "biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    def predict_conll_file(self, conll_filepath: str, batch_size: int = 256) -> List[float]:
        # collect instances
        insts = []
        soc = GCN_reader()
        for ext in soc.sentence_iterator(conll_filepath):
            verb_ind = [int(i == ext.pred_ind) for i in range(len(ext.words))]
            tokens = [Token(t) for t in ext.words]
            if not any(verb_ind):
                continue   # skip extractions without predicate

            ##########################
            result = self.dependency_tree_predictor.predict(sentence=" ".join(ext.words))
            predicted_heads = result["predicted_heads"]
            #########################
            verb_index = verb_ind.index(1)
            #############################################
            adj = {}
            self.traverse_predicted_heads(adj, predicted_heads, verb_index + 1)
            # 有些动词没有关系，防止在后面listfield中出错
            adj[verb_index + 1].append(verb_index + 1)
            ##############################################
            insts.append(self._dataset_reader.text_to_instance(
                tokens, verb_ind, tags=soc.map_tags(ext.tags, one_verb=self._dataset_reader._one_verb), adj=adj))

        # run rerank model and get the scores
        outputs = []
        for batch in range(0, len(insts), batch_size):
            batch = insts[batch:batch + batch_size]
            outputs.extend([p['scores'] for p in self._model.forward_on_instances(batch)])
        return outputs
