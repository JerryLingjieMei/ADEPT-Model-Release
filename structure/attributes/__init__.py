from .bin_attributes import BinAttributes
from .base_attributes import BaseAttributes

_ATTRIBUTES_MAP = {"BIN_ATTRIBUTES": BinAttributes,
                   "BASE_ATTRIBUTES": BaseAttributes}


def build_attributes(cfg):
    return _ATTRIBUTES_MAP[cfg.NAME](cfg)
