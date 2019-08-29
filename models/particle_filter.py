import random
import time
import os
import logging
from copy import deepcopy, copy
from statistics import mean

import pycocotools.mask as mask_util
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pybullet as p
import pybullet_data
import numpy as np

from .match import Matcher
from .step import Stepper
from physics.loss import build_loss
from physics.visualize.mask_visualize import visible_mask, visible_area
from utils.geometry import iou


class _ParticleUpdater(object):
    def __init__(self, cfg, t, belief, camera, observation_history=None):
        self.camera = camera
        self.belief = deepcopy(belief)
        if observation_history is None:
            self.observation_history = [deepcopy(belief)]
        else:
            self.observation_history = copy(observation_history)

        self.n_objects = 0

        self.matcher = Matcher(cfg)
        self.stepper = Stepper(cfg)

        self.t = t
        self.area_threshold = cfg.MODEL.AREA_THRESHOLD

        samples_mass = cfg.MODEL.MASS
        self.to_sample_mass = samples_mass.TO_SAMPLE_MASS
        if self.to_sample_mass:
            self.log_mass_mu = samples_mass.LOG_MASS_MU
            self.log_mass_sigma = samples_mass.LOG_MASS_SIGMA
            new_belief = []
            for obj in self.belief:
                if obj["type"] != "Occluder":
                    obj = self._assign_object_mass(obj)
                new_belief.append(obj)
            self.belief = new_belief

        matched_cfg = cfg.MODEL.UPDATING.MATCHED
        self.matched_loss = build_loss(matched_cfg.LOSS)
        self.matched_penalty = {
            "location": matched_cfg.LOCATION_SIGMA,
            "velocity": matched_cfg.VELOCITY_SIGMA,
            "scale": matched_cfg.SCALE_SIGMA
        }

        unmatched_belief_cfg = cfg.MODEL.UPDATING.UNMATCHED_BELIEF
        self.unmatched_belief_penalty = {
            "base": unmatched_belief_cfg.BASE_PENALTY,
            "mask": unmatched_belief_cfg.MASK_PENALTY,
        }
        self.unmatched_observation_penalty = cfg.MODEL.UPDATING.UNMATCHED_OBSERVATION.PENALTY
        self.unmatched_observation_max_penalty = cfg.MODEL.UPDATING.UNMATCHED_OBSERVATION.MAX_PENALTY

        self.score = 0.
        self.magic_penalty = 0.
        self.object_locations = [[], []]

    def step(self):
        """Belief step"""
        self.belief, magic_penalty = self.stepper.step(self.belief)
        self.magic_penalty += magic_penalty
        self.t += 1

    def update(self, observation):
        """Observation update"""
        self.observation_history.append(observation)
        n_belief_objects, n_observation_objects = len(self.belief), len(observation)
        n = max(n_belief_objects, n_observation_objects) + 1
        match = self.matcher.get_match(self.belief, observation)

        for i in range(n):  # iterate through observations (including null registers)
            matched_belief_id = match[i]
            if i < n_observation_objects:  # observed object
                obj = observation[i]
                if matched_belief_id >= n_belief_objects:
                    if obj["type"] != "Occluder":
                        self._handle_unmatched_observation(obj)
                else:
                    if self.belief[matched_belief_id] == "Occluder":
                        obj["type"] = "Occluder"
                    if obj["type"] != "Occluder":
                        self._handle_matched(obj, matched_belief_id)
                    else:
                        self._handle_occluder(obj, matched_belief_id)
            elif match[i] < n_belief_objects:
                if self.belief[matched_belief_id]["type"] != "Occluder":
                    self._handle_unmatched_belief(matched_belief_id)
                else:
                    observation.append(self.belief[matched_belief_id])

        for obj in self.belief:
            if obj["type"] != "Occluder":
                self.object_locations[0].append(obj["location"][0])
                self.object_locations[1].append(obj["location"][1])

    def step_and_update(self, observation):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        self.step()
        self.update(observation)
        return self

    def _handle_matched(self, observed_obj, match_belief_id):
        """Handle the case when a belief is matched with an observation."""
        match_obj = self.belief[match_belief_id]
        belief_mask = visible_mask(self.belief, self.camera, match_belief_id)
        mask_iou = iou(belief_mask, mask_util.decode(observed_obj["mask"]))
        if mask_iou < .5:
            terms = ["location", "velocity", "scale"]
            for term in terms:
                score = self.matched_loss(match_obj[term], observed_obj[term], self.matched_penalty[term]) / 2
                self.score += score
        self.n_objects += 1

    def _handle_unmatched_belief(self, unmatched_belief_id):
        """Handle the case when a belief is not matched and 'disappeared'."""
        mask_area = visible_area(self.belief, self.camera, unmatched_belief_id)
        if mask_area > self.area_threshold:  # it should have been seen
            score = self.unmatched_belief_penalty["base"] \
                    + self.unmatched_belief_penalty["mask"] * mask_area ** 2
            self.score += score
            self.n_objects += 1

    def _handle_unmatched_observation(self, obj):
        """Handle the case when a observation is not matched and just 'appeared'."""
        obj_state = [o for o in self.belief if o['type'] != 'Occluder']
        obj_state.append(obj)
        magic_penalty = 0
        for t in reversed(range(self.t)):
            occluder_states = [o for o in self.observation_history[t] if o['type'] == 'Occluder']
            obj_state = self.stepper.reverse_step(obj_state)
            all_states = occluder_states + obj_state
            if t < self.t - 10:
                mask_area = visible_area(all_states, self.camera, len(all_states) - 1)
                if mask_area > self.area_threshold:
                    magic_penalty += self.unmatched_observation_penalty * mask_area
            if magic_penalty > self.unmatched_observation_max_penalty:
                break
        self.magic_penalty += magic_penalty

        # pass the retrospective test
        if self.to_sample_mass:
            obj = self._assign_object_mass(obj)
        self.belief.append(obj)

    def _handle_occluder(self, obj, matched_belief_id):
        if obj["scale"][0] < .05:
            self.belief[matched_belief_id] = obj

    def _assign_object_mass(self, obj):
        obj = deepcopy(obj)
        obj["mass"] = np.exp(np.random.normal(self.log_mass_mu, self.log_mass_sigma))
        return obj

    def get_magic_penalty(self):
        return self.magic_penalty

    def get_score(self):
        if self.n_objects == 0:
            return self.magic_penalty
        else:
            return self.score / self.n_objects + self.magic_penalty

    def get_locations(self):
        return self.object_locations


class FilterUpdater(object):
    def __init__(self, cfg, belief, case_name, camera, n_filter):
        self.t = 0
        self.cfg = cfg
        self.belief = deepcopy(belief)
        self.camera = camera
        self.scores = []
        self.raw_scores = []
        self.raw_locations = [[], []]
        self.raw_velocities = [[], []]

        self.n = cfg.MODEL.N_PARTICLES
        self.particle_scores = [0] * self.n
        self.resample_period = cfg.MODEL.RESAMPLE.PERIOD
        self.resample_factor = cfg.MODEL.RESAMPLE.FACTOR

        self.weights = [1 / self.n] * self.n
        self.particles = [_ParticleUpdater(cfg, 0, belief, camera) for _ in range(self.n)]
        self.logger = logging.getLogger("{}{}".format(cfg.LOG_PREFIX, case_name))
        self.n_filter = n_filter

    @staticmethod
    def _combined_step_and_update(particles, observation):
        """Do multiple step and update in one process"""
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        results = []
        for particle in particles:
            results.append(particle.step_and_update(observation))
        p.disconnect()
        return results

    def step_and_update(self, observation):
        """Combination of physics step update and observation update"""
        if self.n_filter.acquire(False):
            worker_args = []
            pool_size = int(cpu_count() // 2)
            for i in range(0, self.n, pool_size):
                worker_args.append((self.particles[i:i + pool_size], observation))
            with Pool(pool_size) as pool:
                particles_lists = pool.starmap(FilterUpdater._combined_step_and_update, worker_args)
            self.particles = []
            for particles_list in particles_lists:
                self.particles.extend(particles_list)
            self.n_filter.release()
        else:
            p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            for i in range(self.n):
                self.particles[i].step_and_update(observation)
            p.disconnect()
        self.t += 1

        likelihood = 0
        raw_scores = []
        raw_locations = [[], []]
        for i in range(self.n):
            score = self.particles[i].get_score()
            raw_scores.append(score)
            self.particle_scores[i] += score
            likelihood += self.weights[i] * np.exp(-score)

            locations = self.particles[i].get_locations()
            for j in range(2):
                raw_locations[j].extend(locations[j])
        self.scores.append(-np.log(likelihood))
        self.raw_scores.append(raw_scores)

        for j in range(2):
            self.raw_locations[j].append(raw_locations[j])

    def resample(self):
        """Resample according to the weight of particles"""
        new_particles = []
        weight_sum = 0
        min_score = min(self.particle_scores)
        for i in range(self.n):
            score = self.particle_scores[i] * self.resample_factor
            likelihood = np.exp(-(score - min_score))
            self.weights[i] = likelihood
            weight_sum += likelihood

        for i in range(self.n):
            self.weights[i] /= weight_sum

        choices = random.choices(self.particles, self.weights, k=self.n)
        for i, choice in enumerate(choices):
            new_particles.append(
                _ParticleUpdater(self.cfg, choice.t, choice.belief, self.camera,
                                 choice.observation_history))
        self.particles = new_particles
        self.particle_scores = [0] * self.n
        self.weights = [1 / self.n] * self.n

    def run(self, observations):
        """feed the updator with observations"""
        for i, observation in enumerate(tqdm(observations)):
            self.step_and_update(observation["objects"])

            log_info = "At time {}\n".format(self.t)
            log_info += "{:>16}{:>16}{:>16}\n".format("object_beliefs", "n_particles", "weight")
            for j in range(5):
                particles_score = list(weight for weight, particle in zip(self.weights, self.particles)
                                       if len(particle.belief) == j)
                log_info += "{:>16}{:>16}{:>16.4f}\n".format(j, len(particles_score), sum(particles_score))
            log_info += "Particles has a average magic penalty of {:.4f}\n".format(
                sum(weight * particle.get_magic_penalty() for weight, particle in zip(self.weights, self.particles)))
            log_info += "The current score is {:.4f}\n".format(self.scores[-1])
            self.logger.info(log_info)

            if self.t % self.resample_period == 0:
                self.resample()

    def get_score(self):
        metrics = [sum, mean, max]
        # noinspection PyCallingNonCallable
        score_dict = {metric.__name__: metric(self.scores) for metric in metrics}
        score_dict["all"] = self.scores
        score_dict["raw"] = self.raw_scores
        score_dict["location"] = self.raw_locations
        return score_dict
