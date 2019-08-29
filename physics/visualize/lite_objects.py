import os

import pybullet as p
import numpy as np

from ..camera import Camera
from utils.constants import COLORS2RGB, SHAPES2TYPES


class LiteObjectManager(object):

    def __init__(self, config, rendering_config, obj_dir):
        p.resetSimulation()
        self.obj_dir = obj_dir
        self.objects = config
        self.camera = self.set_camera(rendering_config)

        self.object_ids = []
        for obj_params in self.objects:
            self.object_ids.append(self.add_object(**obj_params))

        vis_id = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(self.obj_dir, "shapes", 'ground.obj'),
                                     meshScale=[10, 10, 10], rgbaColor=[*(x / 256 for x in [150, 150, 150]), 1])
        self.ground_id = p.createMultiBody(baseVisualShapeIndex=vis_id, basePosition=[0, 0, 0])

    def add_object(self, type, location, rotation, scale, **kwargs):
        '''
        create an bybullet base object from a wavefront .obj file
        set up initial parameters and physical properties
        '''
        shape = SHAPES2TYPES[type]
        color = COLORS2RGB[kwargs["color"]] if "color" in kwargs else COLORS2RGB["green"]
        obj_path = os.path.join(self.obj_dir, "shapes", '%s.obj' % shape)
        orn_quat = p.getQuaternionFromEuler(rotation)
        vis_id = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, meshScale=scale,
                                     rgbaColor=[*(x / 256 for x in color), 1])
        obj_id = p.createMultiBody(baseVisualShapeIndex=vis_id, basePosition=location, baseOrientation=orn_quat)
        return obj_id

    def set_camera(self, rendering):
        return Camera(rendering.camera_look_at, -rendering.camera_theta, rendering.camera_phi + 90, 0,
                      rendering.camera_rho, fov=32, width=480, height=320)

    def get_area(self, object_id):
        """Tells the area of an object"""
        segmentation = self.camera.take_seg()
        return np.sum(np.array(segmentation) == object_id)

    def get_mask(self, object_id):
        """Tells the mask of an object"""
        segmentation = self.camera.take_seg()
        return np.array(segmentation) == object_id

    def take_image(self):
        """Check if an object is visible"""
        image = self.camera.take_pic()[:, :, :3]
        return image
