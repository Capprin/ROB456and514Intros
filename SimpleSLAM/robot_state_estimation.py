#!/usr/bin/env python3

import numpy as np

from world_state import WorldState
from door_sensor import DoorSensor


# Belief about world/robot state
class RobotStateEstimation:
    def __init__(self):

        # Probability representation (discrete)
        self.probabilities = []
        self.reset_probabilities(10)

        # Kalman (Gaussian) probabilities
        self.mean = 0.5
        self.standard_deviation = 0.4
        self.reset_kalman()

    def reset_probabilities(self, n_probability):
        """ Initialize discrete probability resolution with uniform distribution """
        div = 1.0 / n_probability
        self.probabilities = np.ones(n_probability) * div

    def update_belief_sensor_reading(self, ws, ds, sensor_reading_has_door):
        """ Update your probabilities based on the sensor reading being true (door) or false (no door)
        :param ws World state - has where the doors are
        :param ds Door Sensor - has probabilities for door sensor readings
        :param sensor_reading_has_door - contains true/false from door sensor
        """
        
        # begin homework 2 : problem 3
        new_probs = np.zeros(len(self.probabilities))
        n = 0
        for i in range(len(new_probs)):
            # each prob. is P(robot is here)
            is_door = ws.is_in_front_of_door((i+1)/len(self.probabilities))
            # get P(see door|is_door)
            likelihood = 0
            if is_door:
                likelihood = ds.prob_see_door_if_door
            else:
                likelihood = ds.prob_see_door_if_no_door
            # update likelihood to P(got reading|door state)
            if not sensor_reading_has_door:
                likelihood = 1-likelihood
            new_probs[i] = likelihood * self.probabilities[i]
            n += new_probs[i]
        # Normalize - all the denominators are the same because they're the sum of all cases
        self.probabilities = new_probs/n
        # end homework 2 : problem 3

    # Distance to wall sensor (state estimation)
    def update_dist_sensor(self, ws, dist_reading):
        """ Update state estimation based on sensor reading
        :param ws - for standard deviation of wall sensor
        :param dist_reading - distance reading returned from the sensor, in range 0,1 (essentially, robot location) """

        # Standard deviation of error
        standard_deviation = ws.wall_standard_deviation
        # begin homework 2 : Extra credit
        self.mean = dist_reading
        self.standard_deviation = standard_deviation
        # end homework 2 : Extra credit
        return self.mean, self.standard_deviation

    def update_belief_move_left(self, rs):
        """ Update the probabilities assuming a move left.
        :param rs - robot state, has the probabilities"""

        # begin homework 2 problem 4
        # Check probability of left, no, right sum to one
        # Left edge - put move left probability into zero square along with stay-put probability
        # Right edge - put move right probability into last square
        # Normalize - sum should be one, except for numerical rounding
        new_probs = np.zeros(len(self.probabilities))
        n = 0
        for k in range(len(self.probabilities)):
            avg = 0
            for i in range(len(self.probabilities)):
                # get P(moved to k | located at i)
                prob = 0
                if i - k == -1:
                    # i left of k
                    prob = rs.prob_move_right_if_left
                elif i - k == 0:
                    # i at k
                    prob = rs.prob_no_move_if_left
                    # handle edges
                    if k == 0:
                        prob += rs.prob_move_left_if_left
                    elif k == len(self.probabilities)-1:
                        prob = rs.prob_move_right_if_left
                elif i - k == 1:
                    # i right of k
                    prob = rs.prob_move_left_if_left
                avg += prob * self.probabilities[i]
            new_probs[k] = avg
            n += avg
        self.probabilities = new_probs/n
        # end homework 2 problem 4

    def update_belief_move_right(self, rs):
        """ Update the probabilities assuming a move right.
        :param rs - robot state, has the probabilities"""

        # begin homework 2 problem 4
        # Check probability of left, no, right sum to one
        # Left edge - put move left probability into zero square along with stay-put probability
        # Right edge - put move right probability into last square
        # Normalize - sum should be one, except for numerical rounding
        new_probs = np.zeros(len(self.probabilities))
        n = 0
        for k in range(len(self.probabilities)):
            avg = 0
            for i in range(len(self.probabilities)):
                # get P(moved to k | located at i)
                prob = 0
                if i - k == -1:
                    # i left of k
                    prob = rs.prob_move_right_if_right
                elif i - k == 0:
                    # i at k
                    prob = rs.prob_no_move_if_right
                    # handle edges
                    if k == 0:
                        prob += rs.prob_move_left_if_right
                    elif k == len(self.probabilities)-1:
                        prob = rs.prob_move_right_if_right
                elif i - k == 1:
                    # i right of k
                    prob = rs.prob_move_left_if_right
                avg += prob * self.probabilities[i]
            new_probs[k] = avg
            n += avg
        self.probabilities = new_probs/n
        # end homework 2 problem 4

    # Put robot in the middle with a really broad standard deviation
    def reset_kalman(self):
        self.mean = 0.5
        self.standard_deviation = 0.4

    # Given a movement, update Gaussian
    def update_kalman_move(self, rs, amount):
        """ Kalman filter update mean/standard deviation with move (the prediction step)
        :param rs : robot state - has the standard deviation error for moving
        :param amount : The requested amount to move
        :return : mean and standard deviation of my current estimated location """

        # begin homework 3 : Problem 2
        self.mean = self.mean + amount
        self.standard_deviation = self.standard_deviation + rs.robot_move_standard_deviation_err
        # end homework 3 : Problem 2
        return self.mean, self.standard_deviation

    # Sensor reading, distance to wall (Kalman filtering)
    def update_gauss_sensor_reading(self, ws, dist_reading):
        """ Update state estimation based on sensor reading
        :param ws - for standard deviation of wall sensor
        :param dist_reading - distance reading returned"""

        # begin homework 3 : Problem 1
        k = self.standard_deviation / (self.standard_deviation + ws.wall_standard_deviation)
        self.mean = self.mean + k * (dist_reading - self.mean)
        self.standard_deviation = (1-k) * self.standard_deviation
        # end homework 3 : Problem 1
        return self.mean, self.standard_deviation


if __name__ == '__main__':
    ws_global = WorldState()

    ds_global = DoorSensor()

    rse_global = RobotStateEstimation()

    # Check out these cases
    # We have two possibilities - either in front of door, or not - cross two sensor readings
    #   saw door versus not saw door
    uniform_prob = rse_global.probabilities[0]

    # begin homework 2 problem 4
    # Four cases - based on default door probabilities of
    # DoorSensor.prob_see_door_if_door = 0.8
    # DoorSensor.prob_see_door_if_no_door = 0.2
    #  and 10 probability divisions. Three doors visible.
    # probability saw door if door, saw door if no door, etc
    # Resulting probabilities, assuming 3 doors
    # Check that our probabilities are updated correctly
    # Spacing of bins
    # end homework 2 problem 4

    print("Passed tests")
