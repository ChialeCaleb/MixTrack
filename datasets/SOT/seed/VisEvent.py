from datasets.base.factory_seed import BaseSeed
from datasets.types.data_split import DataSplit


class VisEvent_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit = DataSplit.Training | DataSplit.Validation, sequence_filter=None):
        if root_path is None:
            root_path = self.get_path_from_config('VisEvent_PATH')
        self.sequence_filter = sequence_filter
        name = 'VisEvent'
        if sequence_filter is not None:
            name += '-'
            name += sequence_filter
        super(VisEvent_Seed, self).__init__(name, root_path, data_split, 2)

    def construct(self, constructor):
        from .Impl.VisEvent import construct_VisEvent
        construct_VisEvent(constructor, self)
