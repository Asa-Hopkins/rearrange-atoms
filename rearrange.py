# -*- coding: utf-8 -*-
"""
Updated on Thu Nov 23 13:05 2023

Available at https://github.com/Asa-Hopkins/rearrange-atoms/

@author: Asa Hopkins
"""



import numpy as np
from scipy import ndimage
import time
from matplotlib import pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist

def calc_dist_arr(d):
    """Generates the positions that are d steps from a central point

    Parameters
    ----------
    d : int
        The number of steps to take

    Returns
    -------
    dist : numpy array
        An array which is 1 where the coordinate is d steps from the centre
    y - d : numpy array
        The y offsets of the points that are d steps from the centre
    x - d : numpy array
        The x offsets of the points that are d steps from the centre"""
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
    """Finds the parent of an item, and applies path compression

    Parameters
    ----------
    parent : dictionary
        A dictionary mapping each item to its parent
    item : tuple
        A coordinate which we want to find the parent of

    Returns
    -------
    parent[item] : tuple
        The coordinate which is the parent of the given one"""
    if parent[item] != item:
        parent[item] = find(parent,parent[item])
    return parent[item]

def has_unique_neighbors(lw, i):
    """Highlights all positions where the number of surrounding clusters is at least 3

    Parameters
    ----------
    lw : numpy array
        An array where all clusters have been assigned the same value
    i : int
        Which of the four rotations to check

    Returns
    -------
    unique & nonzero : numpy array
        An array which equals 1 where the number of surrounding clusters >=3"""
    
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
    
    # Check if all neighbours are different
    unique = (rot[0] != rot[1]) & (rot[0] != rot[2]) & (rot[1] != rot[2])
    nonzero = (rot[0] != 0) & (rot[1] != 0) & (rot[2] != 0)
    return unique & nonzero

def fill_in(arr, pad = True):
    """Clears sites in the given array to reach percolation (i.e so one cluster remains)
    We swap 0 and 1 to be compatable with numpy functions, so it's actually a fill-in algorithm.
    This is equivalent to another problem called the Rectilinear Steiner Tree, which is NP-hard.
    Using the statistics of the situation, we can still get nearly optimal results in (almost) O(n^2) time.
    Parameters
    ----------
    arr : numpy array
        An array with 1s in filled sites and 0s in empty sites
    pad : bool
        Whether to pad with 0s or not, useful for benchmarking

    Returns
    -------
    arr : numpy array
        An array with 1s in filled sites and 0s in empty sites, with a path between all holes now"""
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
        highest = ndimage.maximum_filter(lw,footprint=dist, mode='constant')
        positions = np.argwhere((highest > lw) & arr)
        
        #We essentially use Kruskal's algorithm to build a MST to connect the remaining clusters

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

cache = [0]

def push_to_bottom(start, sites, inside, bounds):
    """Given an atom in the target array, moves it to the end of the tree
    Parameters
    ----------
    start : tuple
        The coordinate of the atom to be moves
    sites : dictionary
        Maps each coordinate to an array containing its parent and its children coordinates
    inside : numpy array
        An array describing the state of the target zone, with a 1 at filled positions and a 0 at unfilled ones. 

    Returns
    -------
    paths : list
        A list describing the path taken to move the atom from the start to the end of the tree"""
    #takes position relative to bounding box
    #returns path relative to entire array
    global cache
    x1,y1,x2,y2 = bounds
    prev = 0
    pos = tuple(start)
    paths = []
    path = [(pos[0] + y1, pos[1] + x1)]
    inside[pos] = 0
    move = None
    if cache[0] == path[0] and len(cache) > 2:
        pos = cache[-1]
        path = cache[:-1]
        move = cache[-2]
        
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
    paths.append(path)
    cache = path[:-1] + [prev]
    return paths

def move_to_hole(a,b,bounds):
    """Moves an atom from point a to point b, making sure not to cross the target area

    Warning: still has some bugs    
    
    Parameters
    ----------
    a : tuple
        starting coordinate
    b : tuple
        end coordinate
    bounds : list
        describes the rectangle containing the target area

    Returns
    -------
    move : list
        Describes the path taken to move from a to b, avoiding the target area"""
    
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
    
    
def create_moves(atoms, fill, bounds):
    """Takes an array of atom positions and a copy modified by fill_in, and generates
    a set of moves to fill in the remaining target area.
    
    Parameters
    ----------
    atoms : numpy array
        An array with 1s in filled sites and 0s in empty sites
    fill : numpy array
        An array of just the target area with 1s in filled sites and 0s in empty sites
    bounds : list
        describes the rectangle of the target region

    Returns
    -------
    moves : list
        A list describing every move needed to fill in the remainder of the array"""
    #Essentially builds a Minimum Spanning Tree, then pushes atoms to the bottom until it's full
    #Complexity O(n^3)
    #Total path length will be O(n^3), so that's the lower bound

    #This will map each point to a list of lists    
    #first list is the connected point, second is the root point
    #Third is number of points under this one, only needed for the root point
    sites = {}
    
    #find zeros on the periimeter
    perim = fill.copy()
    perim[1:-1,1:-1] = 1

    x1,y1,x2,y2 = bounds

    #Keeps track of perimeter points, plus depth of its tree
    perimeter = {}
    
    #We perform a breadth first search to reach every site in the shortest distance
    #Add unsearched sites to the queue
    q = []
    ind = 0
    for site in np.argwhere(perim == 0):
        site = tuple(site)
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
                sites[site][0].append((adj[0] - site[0], adj[1] - site[1]))
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
            push = push_to_bottom(atom, sites, inside, bounds)
            moves.extend(push)
            root = sites[tuple(atom)][1]
            perimeter[(root[0] + y1, root[1] + x1)] -= len(push)
            
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
    holes = []
    for i in perimeter:
        holes.append(i)
        depth.append(perimeter[i])
    holes = np.array(holes)
    
    #For each hole, find the nearest unassigned atom
    #Occassionally moves atoms through each other - needs fixing
    dists = cdist(holes,free_atoms,'cityblock')
    order = np.argsort(dists)
    assigned = {}
    for hole_num in np.argsort(np.array(depth)):
        hole = tuple(holes[hole_num])
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
    """Given an initial array of atoms, performs the full rearranging

    Parameters
    ----------
    sites : numpy array
        An array with 1s in filled sites and 0s in empty sites
    bounds : list
        describes the rectangle of the target region
    pad : bool
        Whether to pad the array or not, is passed to fill_in

    Returns
    -------
    moves : list
        A list describing every move needed to fill in the remainder of the array
    """
    #Input a binary array representing the initial loading
    #and a bounding box for target site locations
    x1,y1,x2,y2 = bounds
    targets = sites[y1:y2,x1:x2]
    fill = fill_in(targets)
    moves = create_moves(sites,fill,bounds)
    return moves

def random_run(N, bounds , seed , animate = False):
    """Creates a random lattice loading and rearranges it

    Parameters
    ----------
    N : int
        Width of the lattice to generate
    bounds : list
        describes the rectangle of the target region
    seed : int
        The random seed to use, to allow for reproducibility
    animate : bool
        Whether to generate an animation or not

    Returns
    -------
    moves : list
        A list describing every move needed to fill in the remainder of the array"""
    
    np.random.seed(seed)
    arr = np.random.randint(0,2,(N,N))
    copy = arr.copy()
    moves = rearrange(arr,bounds)
    
    if animate:
        frames = []
        fig = plt.figure()
        
        def animate(i):
            #Each frame, we clear the plot, then make a new one
            print(i)
            plt.cla()
            plt.imshow(1 - copy, cmap='gray')
            plt.xlim((-1,N))
            plt.ylim((-1,N))
            nx.draw_networkx(frames[i][1], pos = frames[i][2], with_labels=False, node_size = 5, arrowstyle="->", arrowsize=5, edge_color='red')
            copy[frames[i][0][0]] = 0
            copy[frames[i][0][1]] = 1
            return plt
    
        # Create an empty graph
        for move in moves:
            graph = nx.DiGraph()
            d = {}
            pos = move[0]
            prev = pos
            graph.add_node(pos)
            d[pos] = pos[::-1]
            mov = move[1]
            for i in move[2:]:
                if mov[0]*i[0] + mov[1]*i[1] != 0:
                    mov = [mov[0] + i[0], mov[1] + i[1]]
                else:
                    prev = pos
                    pos = (pos[0] + mov[0], pos[1] + mov[1])
                    mov = i
                    graph.add_node(pos)
                    graph.add_edge(prev,pos)
                    d[pos] = pos[::-1]
            
            prev = pos
            pos = (pos[0] + mov[0], pos[1] + mov[1])
            graph.add_node(pos)
            graph.add_edge(prev,pos)
            d[pos] = pos[::-1]
            frames.append([(move[0],pos),graph,d])
    
        anim = FuncAnimation(fig = fig, func = animate, frames = len(moves), interval = 100, repeat = False)
        anim.save('Animation.gif',fps=5, dpi=100)
    return moves
    
def average(n,reps, anim = False):
    """Used for doing repeated runs to generate statistics."""
    #For now, a square target array of width n is used, and N is calculated
    #such that there are enough spare atoms to do the rearranging
    #Later, more general target shapes will be allowed
    vals = []
    times = []
    paths = []
    N = int(n*1.5 + 1) #Should have enough atoms
    diff = (N - n)//2
    bounds = (diff, diff, diff + n, diff + n)
    for rep in range(0,reps):
        t = time.time()

        moves = random_run(N,bounds,rep,anim)
        #Moves needed to rearrange 
        vals.append(len(moves)/n**2)

        #path length needed to rearrange
        paths.append(sum([np.sum(np.abs(move[1:])) for move in moves]))
        times.append(time.time() - t)
    return vals, paths, times

def graphs():
    """Uses the average function to generate graphs"""
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

average(40,1,True)
