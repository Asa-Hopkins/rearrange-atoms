# rearrange-atoms

## About
Code for performing a rearrangement of a stochastic loading of atoms into a rectangular target area

## Usage
To calculate a set of moves, call `rearrange(sites,bounds)`, where `sites` is a binary array describing the occupied sites, and `bounds` contains four integers like `(y1,x1,y2,x2)` describing the top left and bottom right corners of the target zone. (there's a mistake currently where the code interprets it as `(x1,y1,x2,y2)`)

The function returns an array of moves, in the order they should be executed. These moves are arrays, with the first element describing the coordinate of the atom to pick up,and the following elements describing changes in position from the current position. Note that `numpy` indexes arrays like `pos[y,x]`, where y is measured from the top, so this is the expected layout of the input array.

## Algorithm Description
Given a 2D binary array and a bounding box for the target sites, `rearrange` will first calculate which atoms need to be replaced with holes to connects all the clusters of holes into one single cluster.

Then, a Minimum Spanning Tree is calculated using a breadth first search starting from every hole on the perimeter. Any MST would be fine, this choice of MST should reduce path lengths though. Then, the atoms that we said need to be replaced with holes are pushed to the bottom of the MST (measuring depth starting at the perimeter).

After this, every move takes an atom outside the target zone, and fills a target site, so is optimal from this point in the number of moves. Currently, the shallowest branch of the MST is picked. Suppose it has depth `n`, then a shortest-path algorithm finds the nearest `n` sites and moves them to the bottom of the branch, filling it completely before moving on.

I have tested using an exact linear sum optimiser for the pairing of atoms with sites, but the difference was minimal despite having a `O(N^3)` complexity. The current method is `O(N^(3/2)logN)` as it must sort all the distances after calculating them. This is theoretically the limiting factor, but in reality the calculation of the moves (which is `O(N^(3/2))`) is much slower as it's pure python and not a `numpy`/`scipy` function.

It can be seen that the average distance of a site from the perimeter is `O(N^0.5)` (since `N^0.5` is the length of a side of the square), so the total path length will always be at least `O(N^3/2)`, as `O(N)` points outside the target area will have to be moved at least the distance from the perimeter to the hole. This is therefore a lower bound on the algorithmic complexity
