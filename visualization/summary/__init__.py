from .attribute_reconstruction import AttributeReconstructionVisualizer

_VISUALIZATION_MAP = {
                      "ATTRIBUTE_RECONSTRUCTION": AttributeReconstructionVisualizer}


def build_summary_visualization(name, summary_writer):
    return _VISUALIZATION_MAP[name](summary_writer)
