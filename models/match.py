import itertools

import numpy as np

from physics.loss import l2


class Matcher(object):
    def __init__(self, cfg):
        matcher_cfg = cfg.MODEL.MATCHER
        self.penalty = {
            "type": matcher_cfg.TYPE_PENALTY,
            "distance": matcher_cfg.DISTANCE_PENALTY,
            "color": matcher_cfg.COLOR_PENALTY,
        }
        self.distance_threshold = matcher_cfg.DISTANCE_THRESHOLD
        self.base_penalty = matcher_cfg.BASE_PENALTY

    def _criterion(self, obj_1, obj_2):
        """
        Compute a location_loss between obj_1 and obj_2. Factors considered:
            - object 'type'
            - 3d location
        """
        loss = 0
        if obj_1['type'] != obj_2['type']:
            if not (obj_1["type"] in ["Cube", "Sphere"] and obj_2["type"] in ["Cube", "Sphere"]):
                loss += self.penalty["type"]
        if 'color' in obj_1 and 'color' in obj_2:
            if obj_1['color'] != obj_2['color']:
                loss += self.penalty["color"]
        penalty_location = l2(obj_1['location'], obj_2['location'], self.distance_threshold)
        if penalty_location > 1:
            loss += self.penalty["distance"]
        return loss

    def _get_match_matrix(self, belief, observation):
        n_belief_objects, n_observation_objects = len(belief), len(observation)
        n = max(n_belief_objects, n_observation_objects) + 1
        matrix = [[self.base_penalty] * n for _ in range(n)]
        for obj_id, obj in enumerate(belief):
            for obj_ann_id, obj_ann in enumerate(observation):
                matrix[obj_ann_id][obj_id] = self._criterion(obj, obj_ann)
        for i in range(n_observation_objects, n):
            for j in range(n_belief_objects, n):
                matrix[i][j] = 0  # not-observed and not-in-belief are ok to match
        return matrix

    def get_match(self, belief, observation):
        """
        Do maximum weight matching for nxn matrix
        """
        matrix = self._get_match_matrix(belief, observation)
        n = len(matrix)
        match, match_cost = list(range(n)), sum(matrix[_][_] for _ in range(n))
        for p in itertools.permutations(range(n)):
            cost = sum(matrix[_][p[_]] for _ in range(n))
            if cost < match_cost:
                match, match_cost = p, cost
        return match

    def get_match_matrix(self, belief, observation):
        return self._get_match_matrix(belief, observation)
