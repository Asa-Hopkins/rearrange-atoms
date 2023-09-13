import numpy as np
from scipy import ndimage
import time
from matplotlib import pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist
from numba import jit

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
    #Finds the parent of an item, and applies path compression
    if parent[item] != item:
        parent[item] = find(parent,parent[item])
    return parent[item]

def has_unique_neighbors(lw, i):
    # Define a function to check if three neighbors are different
    above = np.roll(lw, 1, axis=0)
    above[0,:] = 0
    
    below = np.roll(lw, -1, axis=0)
    below[-1,:] = 0
    
    left = np.roll(lw, -1, axis=1)
    left[:,-1] = 0
    
    right = np.roll(lw, 1, axis=1)
    right[:,0] = 0
    
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
    #Start by filling positions where three or four clusters meet
    for rot in range(0,4):
        #Find clusters
        lw, num = ndimage.label(arr)

        #TODO check for redundancies
        threes = has_unique_neighbors(lw,rot)
        arr = threes | arr

    #Update cluster labels
    lw, num = ndimage.label(arr)

    d = 2
    while num != 1:
        #We now connect all clusters that require 1 or more sites added
        #The drop off of these is exponential, d = 3 is rarely reached
        dist, dist_y, dist_x = calc_dist_arr(d)
        
        #Find all positions with a higher numbered cluster at a distance of d
        temp1 = ndimage.maximum_filter(lw,footprint=dist, mode='constant')
        positions = np.argwhere((temp1 > lw) & arr)
        
        #We essentially use Kruskal's algorithm to build a MST to connect the remaining clusters
        #'inspired' by https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/

        parent = np.int32(range(0,num+1))
        size = np.int32(np.ones(num+1))

        #We want to try keeping cluster sizes uniform
        np.random.shuffle(positions)

        for y,x in positions:
            #Need to find which relative position the other clusters are
            ys = dist_y + y
            ys[np.logical_or(height <= ys,ys < 0)] = y
            xs = dist_x + x
            xs[np.logical_or(width <= xs,xs < 0)] = x
            keys = np.nonzero(lw[ys,xs] > lw[y,x])
            for key in keys[0]:
                edge = lw[y,x], lw[ys[key],xs[key]]
                p0, p1 = find(parent,edge[0]), find(parent,edge[1])
                if p0 != p1:
                    #Union the two sets
                    parent[p0] = p1
                    
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
    if pad:
        return 1 - arr[1:-1,1:-1]
    else:
        return 1 - arr

def push_to_bottom(start, sites, inside, bounds):
    #takes position relative to bounding box
    #returns path relative to entire array

    x1,y1,x2,y2 = bounds
    
    pos = tuple(start)
    paths = []
    path = [pos]
    move = None
    
    while sites[pos][0] != []:
        move = sites[pos][0][-1]
        pos = (pos[0] + move[0], pos[1] + move[1])
        path.append(move)
        #If we encounter another atom, push that one first
        if inside[pos] == 1:
            paths = paths + push_to_bottom(pos, sites, inside, bounds)
    if move is not None:
        prev = (pos[0] - move[0], pos[1] - move[1])
        sites[prev][0].pop()
    
    inside[path[0]] = 0
    path[0] = (path[0][0] + y1, path[0][1] + x1)
    paths.append(path)
    return paths

def move_to_hole(a,b,bounds):
    #Move a point from a to b, avoiding the target area
    #Imagine a rectangle with a and b at opposite corners
    #If this doesn't intersect the target area, either corner can be the turning point
    #If one corner is inside the target area, use the other one
    #Otherwise, move to nearest corner and go around
    x1,y1,x2,y2 = bounds
    #Modify b to be one step away from the perimeter
    shift = [int(b[0] == y2 - 1) - int(b[0] == y1),0]
    if shift[0] == 0:
        shift[1] = int(b[1] == x2 - 1) - int(b[1] == x1)
    b = [b[0] + shift[0], b[1] + shift[1]]

    #Get top left and bottom right corners
    rect = [[min(a[0],b[0]),min(a[1],b[1])], [max(a[0],b[0]),max(a[1],b[1])]]

    #The turning/end points, a straight line is travelled between each
    points = []
    move = [tuple(a)]

    #https://stackoverflow.com/questions/306316/determine-if-two-rectangles-overlap-each-other
    intersect = (rect[0][1] < x2 and rect[1][1] >= x1 and rect[0][0] < y2 and rect[1][0] >= y1)
    if not intersect:
        points.append([a[0],b[1]])
    else:
        if y1 <= a[0] < y2 and x1 <= b[1] < x2:
            points.append([b[0],a[1]])
        elif y1 <= b[0] < y2 and x1 <= a[1] < x2:
            points.append([a[0],b[1]])
        else:
            points.append((0,0))
            points.append((0,0))
    points.append(b)
    for point in points:
        diff = (point[0] - a[0],point[1] - a[1])
        if diff != (0,0):
            move.append(diff)
        a = point
    move.append((-shift[0],-shift[1]))
    return move
    
    
def create_moves(atoms, fill, bounds, exact = False):
    #takes an array of atom positions and its filled version
    #generates a set of moves to fill in the inner (n,n) square
    #Essentially builds an MST, then pushes atoms to the bottom until it's full
    #Complexity O(n^3)
    #Total path length will be O(n^3), so that's the lower bound
    
    sites = {}
    
    #find zeros on the periimeter
    temp = fill.copy()
    temp[1:-1,1:-1] = 1

    x1,y1,x2,y2 = bounds

    #Keeps track of perimeter points, plus depth of its tree
    perimeter = {}
    
    #We perform a breadth first search to reach every site in the shortest distance
    #Add unsearched sites to a queue
    q = []
    ind = 0
    for site in np.argwhere(temp == 0):
        site = tuple(site)
        #first list is the connected nodes, second is the root node
        #Third is number of nodes under this one, only needed for the root node
        sites[site] = [[],site]
        q.append(site)
        site = (site[0] + y1, site[1] + x1)
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
                
                root = (root[0] + y1, root[1] + x1)
                perimeter[root] += 1
                q.append(adj)
        ind+=1

    #Next step is to find sites in our list with an atom already present
    #and then push that atom to the bottom of the tree
    
    moves = []
    inside = atoms[y1:y2,x1:x2]
    #highlight where there's an atom that needs moving
    inside = inside ^ fill
    for atom in np.argwhere(inside):
        #If we fail to move it, another atom was encountered
        #and was moved instead, so retry
        if inside[atom[0],atom[1]]:
            temp = push_to_bottom(atom, sites, inside, bounds)
            moves.extend(temp)
            root = sites[tuple(atom)][1]
            perimeter[(root[0] + y1, root[1] + x1)] -= len(temp)
            
    #Finally, connect atoms outside the target zone
    out = np.copy(atoms)
    out[y1:y2,x1:x2] = 0
    
    free_atoms = np.argwhere(out)

    #Do a breadth first search outwards from every point
    #The search automatically gives the path needed

    #Mark target sites so we can avoid them
    out[y1:y2,x1:x2] = 2

    #Start with shallowest first
    depth = []
    temp = []
    for i in perimeter:
        temp.append(i)
        depth.append(perimeter[i])
    temp = np.array(temp)
    
    #For each hole, find the nearest unassigned atom
    dists = cdist(temp,free_atoms,'cityblock')
    order = np.argsort(dists)
    assigned = {}
    for hole_num in np.argsort(np.array(depth)):
        hole = tuple(temp[hole_num])
        i = -1
        while perimeter[hole] > 0:
            i += 1
            if order[hole_num][i] in assigned:
                continue
            move = move_to_hole(free_atoms[order[hole_num][i]],hole,bounds) + push_to_bottom([hole[0] - y1, hole[1] - x1],sites,inside,bounds)[0][1:]
            perimeter[hole] -= 1
            moves.append(move)
            assigned[order[hole_num][i]] = 1
    return moves

def rearrange(sites,bounds,pad = True):
    #Input a binary array representing the initial loading
    #and a bounding box for target site locations
    x1,y1,x2,y2 = bounds
    targets = sites[y1:y2,x1:x2]
    fill = fill_in(targets)
    moves = create_moves(sites,fill,bounds, exact = False)
    return moves

def random_run(n, N, bounds ,seed , animate = False):
    #Create a random lattice loading and rearrange it
    np.random.seed(seed)
    arr = np.random.randint(0,2,(N,N))
    copy = arr.copy()
    moves = rearrange(arr,bounds)
    
    #Number of moves needed is equal to the number of 1s in the central n*n square
    #We calculate this by calculating the number of 0s and subtract from n**2

    if animate:
        frames = []
        fig = plt.figure()
        
        def animate(i):
            #Each frame, we clear the plot, then make a new one
            print(i)
            plt.cla()
            plt.imshow(copy)
            nx.draw_networkx(frames[i][1], pos = frames[i][2], with_labels=False, node_size = 5, arrowstyle="->", arrowsize=5)
            copy[frames[i][0][0]] = 0
            copy[frames[i][0][1]] = 1
            return plt
    
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
    
        anim = FuncAnimation(fig = fig, func = animate, frames = len(moves), interval = 100, repeat = False)
        anim.save('A.mp4',fps=5)
    return moves
    
def average(n,reps):
    vals = []
    times = []
    paths = []
    N = int(n*1.5 + 1) #Should have enough atoms
    diff = (N - n)//2
    bounds = (diff, diff, diff + n, diff + n)
    for rep in range(0,reps):
        t = time.time()

        moves = random_run(n,N,bounds,rep,animate = False)
        #Moves needed
        vals.append(len(moves)/n**2)

        #path length needed
        paths.append(sum([np.sum(np.abs(move[1:])) for move in moves]))
        times.append(time.time() - t)
    return vals, paths, times

def graphs():

    num = np.array([15,25,50,100,200,300])
    reps = np.array([1000,500,100,20,5,3])

    moves = []
    std_moves = []
    path_length = []
    std_path = []
    times = []
    std_times = []
    for i in range(len(num)):
        a,b,c = average(num[i],reps[i])
        moves.append(np.mean(a))
        std_moves.append(np.std(a)/reps[i]**0.5)
        
        path_length.append(np.mean(b))
        std_path.append(np.std(b)/reps[i]**0.5)
        
        
        times.append(np.mean(c))
        std_times.append(np.std(c)/reps[i]**0.5)
    
    plt.errorbar(num**2,moves,std_moves)
    plt.title("Percolation based algorithm - large N")
    plt.xlabel("Number of target sites")
    plt.ylabel("Number of moves needed per site")
    plt.show()

    
    plt.errorbar(num**2,path_length,std_path)
    plt.title("Percolation based algorithm - large N")
    plt.xlabel("Number of target sites")
    plt.ylabel("Length of path travelled")
    plt.show()

    plt.errorbar(num**2,times,std_times)
    plt.title("Percolation based algorithm - large N")
    plt.xlabel("Number of target sites")
    plt.ylabel("Execution time")
    plt.show()

#import pprofile
#prof = pprofile.Profile()
#with prof():
#    temp = average(100,1)
#prof.dump_stats('/tmp/prof.txt')
    
temp = average(100,2)
print(np.mean(temp,axis = 1), np.max(temp, axis = 1))
