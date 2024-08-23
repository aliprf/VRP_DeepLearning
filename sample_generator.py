import random
import math
import os
from tqdm import tqdm


class VRPGenerator:
    def __init__(self, num_files, max_loads, max_distance=720, coord_range=(-500, 500)):
        self.num_files = num_files
        self.max_loads = max_loads
        self.max_distance = max_distance
        self.coord_range = coord_range

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def is_route_valid(self, pickup, dropoff):
        depot = (0, 0)
        load_distance = (self.calculate_distance(depot, pickup) +
                         self.calculate_distance(pickup, dropoff) +
                         self.calculate_distance(dropoff, depot))
        if load_distance <= self.max_distance:
            return True
        return False

    def generate_random_point(self):
        return (random.uniform(*self.coord_range), random.uniform(*self.coord_range))

    def create_vrp_file(self, filename):
        load_number = 1
        loads = []

        random_max_load = random.randint(10, self.max_loads-1)
        while load_number <= random_max_load:
            pickup = self.generate_random_point()
            dropoff = self.generate_random_point()
            if self.is_route_valid(pickup, dropoff):
                loads.append((load_number, pickup, dropoff))
                load_number += 1

        with open(filename, 'w') as f:
            # Write header
            f.write("loadNumber pickup dropoff\n")
            # Write load data
            for load in loads:
                id = load[0]
                pikup = load[1]
                dropoff = load[2]
                f.write(f"{id} ({pikup[0]},{pikup[1]}) ({dropoff[0]},{dropoff[1]})\n")

    def create_multiple_vrp_files(self, directory='vrp_files'):
        """Create multiple VRP files."""
        os.makedirs(directory, exist_ok=True)
        for i in tqdm(range(1, self.num_files + 1)):
            filename = f'{directory}/vrp_file_{i}.txt'
            print('creating=>' + str(filename))
            self.create_vrp_file(filename)
        print(f"{self.num_files} VRP files created in '{directory}' directory.")