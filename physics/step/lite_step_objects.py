import pybullet as p
import pybullet_data
import os
from utils.geometry import reverse_xyz, reverse_euler
from utils.constants import SHAPES2TYPES


class LiteObjectStepManager(object):
    def __init__(self, config, obj_dir, forward=True):
        p.resetSimulation()
        self.obj_dir = obj_dir
        self.objects = config
        self.forward = forward

        self.object_ids = []
        self.types = []
        self.scales = []
        self.colors = []

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        for obj_params in self.objects:
            self.object_ids.append(self.add_object(**obj_params))

        plane_id = p.createCollisionShape(p.GEOM_MESH, fileName="plane.obj", meshScale=[10, 10, 10])
        self.ground_id = p.createMultiBody(0., baseCollisionShapeIndex=plane_id,
                                           basePosition=[0, 0, 0])

    def add_object(self, type, location, rotation, scale, velocity, angular_velocity, color, mass=1,
                   lat_fric=0., restitution=.3, lin_damp=0, angular_damp=0, **kwargs):
        """create an pybullet base object from a wavefront .obj file
        set up initial parameters and physical properties"""
        # Occluders are not put into physical simulation
        mass = scale[0] * scale[1] * scale[2] * 100 if type != "Occluder" else scale[0] * scale[1] * scale[2] * 1000
        self.types.append(type)
        self.scales.append(scale)
        self.colors.append(color)
        type = SHAPES2TYPES[type]
        obj_path = os.path.join(self.obj_dir, "shapes", '%s.obj' % type)
        orn_quat = p.getQuaternionFromEuler(rotation)
        col_id = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=scale)
        obj_id = p.createMultiBody(mass, col_id, basePosition=location, baseOrientation=orn_quat)
        p.changeDynamics(obj_id, -1, lateralFriction=lat_fric, restitution=restitution, linearDamping=lin_damp,
                         angularDamping=angular_damp)
        if self.forward:
            omega_quat = p.getQuaternionFromEuler(angular_velocity)
            p.resetBaseVelocity(obj_id, velocity, omega_quat)
        else:
            omega_quat = p.getQuaternionFromEuler(angular_velocity)
            p.resetBaseVelocity(obj_id, reverse_xyz(velocity), reverse_euler(omega_quat))
        return obj_id

    def get_object_motion(self, obj_id):
        """Return the location, orientation, velocity and angular velocity of an object"""
        loc, quat = p.getBasePositionAndOrientation(obj_id)
        orn = p.getEulerFromQuaternion(quat)
        v, omega = p.getBaseVelocity(obj_id)
        if self.forward:
            object_dict = dict(type=self.types[obj_id], scale=self.scales[obj_id], location=list(loc),
                               rotation=list(orn), velocity=list(v),
                               angular_velocity=list(omega), color=self.colors[obj_id])
        else:
            object_dict = dict(type=self.types[obj_id], scale=self.scales[obj_id], location=list(loc),
                               rotation=list(orn), velocity=reverse_xyz(v),
                               angular_velocity=reverse_euler(list(omega)),
                               color=self.colors[obj_id])
        return object_dict
