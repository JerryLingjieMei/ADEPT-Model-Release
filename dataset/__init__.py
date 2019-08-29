from .object_proposal_dataset import ObjectProposalDataset

_DATASET_NAMES = {"OBJECT_PROPOSAL": ObjectProposalDataset}


def build_dataset(dataset_cfg, args):
    dataset = _DATASET_NAMES[dataset_cfg["NAME"]]
    return dataset(dataset_cfg, args)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
