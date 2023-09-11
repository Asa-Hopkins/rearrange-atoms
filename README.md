# rearrange-atoms
Code for performing a rearrangement of a stochastic loading of atoms into a rectangular target area

Given a 2D binary array and a bounding box for the target sites, `rearange` (I've only just seen the mispelling) will first calculate which atoms need to be replaced with holes to connects all the clusters of holes into one single cluster.

Then, a Minimum Spanning Tree is calculated using a breadth first search starting from every hole on the perimeter. Any MST would be fine, this choice of MST should reduce path lengths though. Then, the atoms that we said need to be replaced with holes are pushed to the bottom of the MST (measuring depth starting at the perimeter).

After this, every move takes an atom outside the target zone, and fills a target site, so is optimal from this point in the number of moves. Currently, the shallowest branch of the MST is picked. Suppose it has depth `n`, then a shortest-path algorithm finds the nearest `n` sites and moves them to the bottom of the branch, filling it completely before moving on.

This is quite far from optimal in terms of path lengths, I will try implementing some other methods to see what better options exist. It is essentially a minimum assignment problem, which can be solved exactly in `O(N^3)` where `N` is the number of pairings to be made (so the number of unfilled sites). The current method is `O(N^(3/2))`, which is already the bottleneck of the algorithm, but the exact solver is well written enough that it might still be useful for larger N.

It can be seen that the average distance of a site from the perimeter is `O(N^0.5)` (since `N^0.5` is the length of a side of the square), so the total path length will always be `O(N^3/2)`, as `O(N)` points outside the target area will have to be moved at least the distance from the perimeter to the hole. This is therefore a lower bound on the algorithmic complexity, which my method seems to achieve, although I haven't done a rigorous analysis yet.
