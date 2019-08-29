from .derender import Derender

_META_ARCHITECTURES = {
    "DERENDER": Derender,
}


def build_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
