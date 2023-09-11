import numpy as np
from scipy import ndimage
import time
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment as lsa

#We use an (N,N) grid, where an inner (n,n) grid is the target
#so number of target sites is n^2
N = 25
n = 15

def calc_dist_arr(d):
    #Generates array of positions that are a distance of d from the centre
    dist = []
    l = 2*d + 1
    for i in range(0,l + 1):
        dist.append(np.zeros(l))
        dist[-1][d - i] = 1
        dist[-1][(d + i) % l] = 1
    dist.pop(d)
    dist = np.array(dist)
    y,x = np.where(dist)
    return dist, y - d, x - d

            
def find(parent,item):
    #Finds the parent of an item
    #Uses the "path halving" technique
    #See https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    while parent[item] != item:
        temp = parent[parent[item]]
        parent[item] = temp
        item = temp
    return item

def has_unique_neighbors(lw, i):
    # Define a function to check if three neighbors are different
    #TODO: avoid the calculation of the unneeded direction

    above = np.roll(lw, 1, axis=0)
    below = np.roll(lw, -1, axis=0)
    left = np.roll(lw, -1, axis=1)
    right = np.roll(lw, 1, axis=1)
    
    rot = [above, below, left, right]
    rot = rot[i:] + rot[:i]
    
    # Check if all neighbors are different
    temp1 = (rot[0] != rot[1]) & (rot[0] != rot[2]) & (rot[1] != rot[2])
    temp2 = (rot[0] != 0) & (rot[1] != 0) & (rot[2] != 0)
    return temp1 & temp2

def fill_in(arr, pad = True):
    #Clears sites in the given array near-optimally to reach percolation
    #We swap 0 and 1 so it's a fill-in algorithm
    #Equivalent to constructive a Rectilinear Steiner Tree between the 1s, but by
    #using the statistics of the situation, we can generate very good solutions
    #Basically O(n^2), set unions have an inverse ackermann function but we can ignore that
    if pad:
        #Pad outside with zeros, but we want it to be ones, so we swap 0 and 1
        arr = np.pad(arr,[(1,1),(1,1)])
        arr = 1-arr
    height, width = arr.shape

    #The idea here is to connect all clusters as optimally as possible
    #Start by filling positions where three clusters meet
    #This covers positions where four meet too
    for rot in range(0,4):
        #Find clusters
        lw, num = ndimage.label(arr)

        #Use the function to find where three clusters converge, and add those sites
        #TODO find spots where four clusters can be connected by placing two sites
        threes = has_unique_neighbors(lw,rot)
        arr = threes | arr

    #Update cluster labels
    #TODO could use a flood fill algorithm to update only the necessary part
    lw, num = ndimage.label(arr)

    d = 2
    while num != 1:
        #We now connect all clusters that require 1 or more sites added
        #The drop off of these is exponential, d = 3 is rarely reached
        #Probably faster to do a shortest path algorithm from
        #each remaining cluster at that point
        dist, dist_y, dist_x = calc_dist_arr(d)
        
        #Find all positions with a higher numbered cluster at a distance of 2
        temp1 = ndimage.maximum_filter(lw,footprint=dist, mode='constant')
        positions = np.argwhere((temp1 > lw) & arr)
        
        #We essentially use Kruskal's algorithm to build a MST to connect the remaining clusters
        #'inspired' by https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/

        parent = np.int32(range(0,num+1))
        size = np.int32(np.ones(num+1))

        for y,x in positions:
            #Need to find which relative position the other clusters are
            ys = (dist_y + y) % height
            xs = (dist_x + x) % width
            keys = np.nonzero(lw[ys,xs] > lw[y,x])
            for key in keys[0]:
                edge = lw[y,x], lw[ys[key],xs[key]]
                p0, p1 = find(parent,edge[0]), find(parent,edge[1])
                if p0 != p1:
                    #Perform union by size to give essentially O(n) complexity
                    if size[p0] >= size[p1]:
                        parent[p1] = p0
                        size[p0] += size[p1]
                    else:
                        parent[p0] = p1
                        size[p1] += size[p0]
                    
                    offset = [0, 0]
                    point = [dist_y[key], dist_x[key]]
                    while abs(offset[0] - point[0]) != 0:
                        #Step along y axis until in line with point
                        offset[0] += np.sign(point[0])
                        arr[y+offset[0], x+offset[1]] = 1
                        
                    while abs(offset[1] - point[1]) != 0:
                        #Step along x axis until in line with point
                        offset[1] += np.sign(point[1])
                        arr[y+offset[0], x+offset[1]] = 1
                    #print(f"connected {edge[0]} and {edge[1]}")
        
        lw, num = ndimage.label(arr)
        d += 1
    return 1 - arr[1:-1,1:-1]

def create_moves(atoms, fill, bounds, exact = False):
    #takes an array of atom positions and its filled version
    #generates a set of moves to fill in the inner (n,n) square
    #Essentially builds an MST, then pushes atoms to the bottom until it's full
    #Complexity is O(n^6) for exact, I think an O(n^3) approximation is possible
    #Total path length will be O(n^3), so that's the lower bound anyway
    
    sites = {}
    
    #find zeros on the periimeter
    temp = fill.copy()
    temp[1:-1,1:-1] = 1

    x1,y1,x2,y2 = bounds

    #Keeps track of perimeter points, plus multiplicity
    if exact:
        perimeter = []
    else:
        perimeter = {}
    
    #We perform a breadth first search to reach every site in the shortest distance
    #Add unsearched sites to a queue
    q = []
    ind = 0
    for site in np.argwhere(temp == 0):
        site = tuple(site)
        #first list is the connected nodes, second is the root node
        #Third is number of nodes under this one, only needed for the root node
        sites[site] = [[],site,1]
        q.append(site)
        site = (site[0] + y1, site[1] + x1)
        if exact:
            perimeter.append(site)
        else:
            perimeter[site] = 1
    
    offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    on_perimeter = True

    while ind < len(q):
        site = q[ind]
        #The four neighbouring sites
        #indices = site + offsets
        indices = [(site[0] + a[0], site[1] + a[1]) for a in offsets]
        
        #Make sure we stay in bounds
        if on_perimeter:
            valid = [(ni, nj) for ni, nj in indices if 0 <= ni < fill.shape[0] and 0 <= nj < fill.shape[1]]
            on_perimeter = len(valid) != 4
        else:
            valid = indices
        
        for adj in valid:
            if adj not in sites and fill[adj] == 0:
                sites[site][0].append(np.array((adj[0] - site[0], adj[1] - site[1])))
                root = sites[site][1]
                sites[adj] = ([],root)
                
                sites[root][2] += 1
                if exact:
                    perimeter.append(root)
                else:
                    root = (root[0] + y1, root[1] + x1)
                    perimeter[root] += 1
                q.append(adj)
        ind+=1

    #Next step is to find sites in our list with an atom already present
    #and then push that atom to the bottom of the tree
    def push_to_bottom(start):
        #takes position relative to bounding box
        #returns path relative to entire array
        
        pos = tuple(start)
        path = [pos]
        move = None
        
        while sites[pos][0] != []:
            move = sites[pos][0][-1]
            pos = (pos[0] + move[0], pos[1] + move[1])
            path.append(move)
            #If we encounter another atom, push that one first
            if inside[pos] == 1:
                moves.append(push_to_bottom(pos))
        if move is not None:
            prev = (pos[0] - move[0], pos[1] - move[1])
            #Remove the site we just filled from tree
            sites[prev][0].pop()
        root = sites[pos][1]
        
        sites[root][2] -= 1
        inside[path[0]] = 0
        path[0] = (path[0][0] + y1, path[0][1] + x1)
        atoms[path[0]] = 0
        atoms[(pos[0] + y1, pos[1] + x1)] = 1
        if not exact:
            perimeter[(root[0] + y1, root[1] + x1)] -= 1
        
        return path
    
    moves = []
    inside = atoms[y1:y2,x1:x2]
    #highlight where there's an atom that needs moving
    inside = inside ^ fill
    for atom in np.argwhere(inside):
        #If we fail to move it, another atom was encountered
        #and was moved instead, so retry
        if inside[*atom]:
            moves.append(push_to_bottom(atom))
    #Finally, connect atoms outside the target zone
    #to perimeter points, using a linear sum assignment optimiser
    out = np.copy(atoms)
    out[y1:y2,x1:x2] = 0
    
    free_atoms = np.argwhere(out)
    
    if exact:
        costs = cdist(free_atoms, np.array(perimeter) + (y1,x1))
        pairs = lsa(costs)
        print(costs[pairs].sum())
    
    else:
        #Do a breadth first search outwards from every point
        #The search automatically gives the path needed

        #Mark target sites so we can avoid them
        out[y1:y2,x1:x2] = 2
        
        for i in perimeter:
            on_perimeter = False
            ind = 0
            q2 = [i]
            visited = {i:[]}
            while ind < len(q2):
                site = q2[ind]
                
                if out[site] == 2 and ind != 0:
                    ind += 1
                    #Stop whenever we hit a target site
                    continue
                
                if out[site] == 1:
                    #If we've found a 1, move it to the root node
                    atoms[site] = 0
                    out[site] = 0
                    atoms[i] = 1
                    path = [site] + visited[site][::-1] + push_to_bottom((i[0] - y1,i[1]-x1))[1:]
                    moves.append(path)

                if perimeter[i] <= 0:
                    break
                
                #The four neighbouring sites
                indices = [(site[0] + a[0], site[1] + a[1]) for a in offsets]
                
                #Make sure we stay in bounds
                valid = [(ni, nj) for ni, nj in indices if 0 <= ni < out.shape[0] and 0 <= nj < out.shape[1]]
                
                for adj in valid:
                    if adj not in visited:
                        move = (site[0] - adj[0], site[1] - adj[1])
                        visited[adj] = visited[site] + [move]
                        q2.append(adj)
                ind+=1
    return moves

def average(n,reps):
    vals = []
    times = []
    paths = []
    for test in range(0,reps):
        t = time.time()
        #Create lattice
        np.random.seed(test)
        N = int(n*1.5 + 1) #Should have enough atoms
        arr = np.random.randint(0,2,(N,N))
        copy = arr.copy()
        diff = (N - n)//2
        bound = (diff, diff, diff + n, diff + n)

        moves = rearange(arr,bound)
        
        #Number of moves needed is equal to the number of 1s in the central n*n square
        #We calculate this by calculating the number of 0s and subtract from n**2

        frames = []
        fig = plt.figure()
        def animate(i):
            #Each frame, we clear the plot, then make a new one
            print(i)
            plt.cla()
            plt.imshow(copy)
            nx.draw_networkx(frames[i][1], pos = frames[i][2], with_labels=False, node_size = 15, arrowstyle="->", arrowsize=10)
            copy[frames[i][0][0]] = 0
            copy[frames[i][0][1]] = 1
            return plt

        if reps == 1:
            # Create an empty graph
            for move in moves:
                graph = nx.DiGraph()
                d = {}
                pos = move[0]
                #plt.imshow(copy)
                prev = pos
                graph.add_node(pos)
                d[pos] = pos[::-1]
                for i in move[1:]:
                    prev = pos
                    pos = (pos[0] + i[0], pos[1] + i[1])
                    graph.add_node(pos)
                    graph.add_edge(prev,pos)
                    d[pos] = pos[::-1]
                frames.append([(move[0],pos),graph,d])
                #Display the problem with solution overlayed
                #nx.draw_networkx(graph, pos = d, with_labels=False, node_size = 10, arrowstyle="->", arrowsize=10)
                #plt.show()
        
            anim = FuncAnimation(fig = fig, func = animate, frames = len(moves), interval = 100, repeat = False)
            anim.save('A.mp4',fps=5)

        #Moves needed
        vals.append(len(moves)/n**2)

        #path length needed
        paths.append(sum([len(move) for move in moves]) - len(moves))
        times.append(time.time() - t)
    print(np.mean(vals), np.std(vals), np.max(vals))
    print(np.mean(paths), np.std(paths), np.max(paths))
    print(np.mean(times), np.std(times), np.max(times))

def rearange(sites,bounds):
    #Input a binary array representing the initial loading
    #and a bounding box for target site locations
    x1,y1,x2,y2 = bounds
    targets = sites[y1:y2,x1:x2]
    fill = fill_in(targets)
    moves = create_moves(sites,fill,bounds, exact = False)
    return moves

#import pprofile
#profiler = pprofile.Profile()
#with profiler:
#    average(15,10)
# Process profile content: generate a cachegrind file and send it to user.

# You can also write the result to the console:
#profiler.dump_stats("/tmp/profiler_stats.txt")

from matplotlib import pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

average(15,1)
