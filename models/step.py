from copy import deepcopy
import numpy as np
import math

from physics.step.step_sim import step, reverse_step


class Stepper(object):
    def __init__(self, cfg):
        step_cfg = cfg.MODEL.STEP
        perturbation_cfg = step_cfg.PERTURBATION
        self.to_perturb = perturbation_cfg.TO_PERTURB
        if self.to_perturb:
            self.velocity_sigma = perturbation_cfg.VELOCITY_SIGMA
            self.location_sigma = perturbation_cfg.LOCATION_SIGMA
            self.scale_sigma = perturbation_cfg.SCALE_SIGMA
            self.velocity_lambda = perturbation_cfg.VELOCITY_LAMBDA
        magic_cfg = step_cfg.MAGIC
        self.use_magic = magic_cfg.USE_MAGIC
        if self.use_magic:
            self.disappear_probability = magic_cfg.DISAPPEAR_PROBABILITY
            self.disappear_penalty = magic_cfg.DISAPPEAR_PENALTY
            self.stop_probability = magic_cfg.STOP_PROBABILITY
            self.stop_penalty = magic_cfg.STOP_PENALTY
            self.accelerate_probability = magic_cfg.ACCELERATE_PROBABILITY
            self.accelerate_penalty = magic_cfg.ACCELERATE_PENALTY
            self.accelerate_lambda = magic_cfg.ACCELERATE_LAMBDA

    def _perturb(self, objects):
        new_objects = deepcopy(objects)
        for obj in new_objects:
            if obj["type"] == "Occluder":
                obj['velocity'] = [0, 0, 0]
                continue
            velocity = math.sqrt(obj['velocity'][0] ** 2 + obj['velocity'][1] ** 2)
            for i in range(2):
                obj['location'][i] += np.random.normal(scale=self.location_sigma[i])
                obj['velocity'][i] += np.random.normal(scale=self.velocity_lambda[i]) * velocity
                obj['velocity'][i] += np.random.normal(scale=self.velocity_sigma[i])
            for i in range(3):
                obj['scale'][i] *= 1 + np.random.normal(scale=self.scale_sigma)
            obj['velocity'][2] = 0
            obj["angular_velocity"] = [0, 0, 0]
        return new_objects

    def step(self, objects):
        if self.use_magic:
            objects, magic_penalty = self._apply_magic(objects)
        else:
            magic_penalty = 0
        if self.to_perturb:
            objects = self._perturb(objects)
        objects = step(objects)
        return objects, magic_penalty

    def reverse_step(self, objects):
        if self.use_magic:
            objects, _ = self._apply_magic(objects, only_stop=True)
        if self.to_perturb:
            objects = self._perturb(objects)
        objects = reverse_step(objects)
        return objects

    def _apply_magic(self, objects, only_stop=False):
        new_objects = []
        magic_penalty = 0
        for obj in objects:
            obj = deepcopy(obj)
            if obj["type"] != "Occluder":
                rand = np.random.rand()
                if not only_stop:
                    if rand < self.disappear_probability:
                        magic_penalty += self.disappear_penalty
                        continue
                    elif rand > 1 - self.stop_probability - self.accelerate_probability:
                        magic_penalty += self.accelerate_penalty
                        velocity = math.sqrt(obj['velocity'][0] ** 2 + obj['velocity'][1] ** 2) + 0.001
                        sigma = math.sqrt(self.velocity_sigma[0] ** 2 + self.velocity_sigma[1] ** 2) + 0.001
                        new_v = np.random.normal(self.accelerate_lambda)
                        for i in range(2):
                            obj["velocity"][i] *= new_v * self.velocity_sigma[i] / velocity / sigma
                if rand > 1 - self.stop_probability:
                    magic_penalty += self.stop_penalty
                    obj["velocity"] = [0, 0, 0]
                if obj['location'][0] < -2 or obj['location'][0] > .5:
                    magic_penalty += 1000
            new_objects.append(obj)
        return new_objects, magic_penalty
