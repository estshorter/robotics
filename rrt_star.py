import math

import matplotlib.pyplot as plt

from rrt import RRT

show_animation = True

"""
Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)
author: AtsushiSakai(@Atsushi_twi)

The MIT License (MIT)

Copyright (c) 2016 - 2021 Atsushi Sakai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class RRTStar(RRT):
    class Node(RRT.Node):
        def __init__(self, x, y) -> None:
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(
        self,
        start,
        goal,
        obstacle_list,
        rand_area,
        expand_dis=30.0,
        path_resolution=1.0,
        goal_sample_rate=20,
        max_iter=300,
        connect_circle_dist=50.0,
        search_until_max_iter=True,
    ):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        super().__init__(
            start,
            goal,
            obstacle_list,
            rand_area,
            expand_dis,
            path_resolution,
            goal_sample_rate,
            max_iter,
        )
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            new_node.cost = nearest_node.cost + math.hypot(
                new_node.x - nearest_node.x + new_node.y - nearest_node.y
            )

            if not self.collide_with_obstacles(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)
            if animation:
                self.draw_graph(rnd_node, i)

            if (not self.search_until_max_iter) and new_node:
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index), i
        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index), self.max_iter - 1
        return None, None

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        costs = []
        for near_ind in near_inds:
            near_node = self.node_list[near_ind]
            t_node = self.steer(near_node, new_node)
            if t_node and not self.collide_with_obstacles(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost
        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        # ゴールに近いノードのインデックス
        goal_inds = [
            dist_to_goal_list.index(dist_to_goal)
            for dist_to_goal in dist_to_goal_list
            if dist_to_goal <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if not self.collide_with_obstacles(t_node, self.obstacle_list):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min(
            [self.node_list[safe_goal_ind].cost for safe_goal_ind in safe_goal_inds]
        )
        for safe_goal_ind in safe_goal_inds:
            if self.node_list[safe_goal_ind].cost == min_cost:
                return safe_goal_ind

        return None

    def find_near_nodes(self, new_node):
        num_node = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(num_node) / num_node))
        if hasattr(self, "expand_dis"):
            r = min(r, self.expand_dis)
        dist_list = [
            (node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2
            for node in self.node_list
        ]
        r_sqr = r ** 2
        near_inds = [dist_list.index(dist) for dist in dist_list if dist <= r_sqr]
        return near_inds

    def rewire(self, new_node, near_inds):
        for near_ind in near_inds:
            near_node = self.node_list[near_ind]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = not self.collide_with_obstacles(
                edge_node, self.obstacle_list
            )
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
        (6, 12, 1),
    ]  # [x, y, radius]
    # Set Initial parameters
    max_iter = 1000
    rrt_star = RRTStar(
        start=[0, 0],
        goal=[6, 10],
        rand_area=[-2, 15],
        obstacle_list=obstacle_list,
        expand_dis=1,
        max_iter=max_iter,
    )
    path, num_iter = rrt_star.planning(animation=False)

    if path is None:
        print("cannnot find path")
    else:
        print("path found")

        if show_animation:
            rrt_star.draw_graph(num_iter=num_iter)
            plt.plot([x for (x, _) in path], [y for (_, y) in path], "-r")
            plt.pause(0.1)
            # plt.grid(True)
            plt.show()


if __name__ == "__main__":
    main()
