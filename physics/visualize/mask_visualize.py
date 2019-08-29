import os
from .lite_objects import LiteObjectManager
from utils.constants import CONTENT_FOLDER


def visible_area(objects, rendering_config, object_id):
    om = LiteObjectManager(objects, rendering_config, os.path.join(CONTENT_FOLDER, "physics/data"))
    visible_area = om.get_area(object_id)
    return visible_area


def visible_mask(objects, rendering_config, object_id):
    om = LiteObjectManager(objects, rendering_config, os.path.join(CONTENT_FOLDER, "physics/data"))
    visible_mask = om.get_mask(object_id)
    return visible_mask


def visible_image(objects, rendering_config):
    om = LiteObjectManager(objects, rendering_config, os.path.join(CONTENT_FOLDER, "physics/data"))
    image = om.take_image()
    return image
