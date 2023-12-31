import csv
import heapq
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar_time(start, end):
    # Begin your code (Part 6)
    """
    Define new heuristic f(x) = g(x) + h(x)
    g(x): Time cost from start to current point
    h(x): (Direct line distance of x and end point)/(edge speed limit)
    Use this heuristic function do A* search.
    """
    edge = {}
    heuristic = {1079387396:{},1737223506:{},8513026827:{}}
    with open(edgeFile) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            st = int(row[0])
            ed = int(row[1])
            distance = float(row[2])
            speed_limit = float(row[3])
            if st not in edge:
                edge[st] = []
            edge[st].append((ed, distance, speed_limit))
    with open(heuristicFile) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pos = int(row[0])
            h1 = float(row[1])
            h2 = float(row[2])
            h3 = float(row[3])
            heuristic[1079387396][pos] = h1
            heuristic[1737223506][pos] = h2
            heuristic[8513026827][pos] = h3

    q = [(heuristic[end][start], start)]
    vis = set()
    time = {}
    par = {}
    time[start] = 0
    vis.add(start)
    while q:
        f, x = heapq.heappop(q)
        if x == end:
            break
        if x not in edge:
            continue
        for y, d, sp in edge[x]:
            t_y = (d/sp)*3.6
            if y not in time or time[y] > t_y + time[x]:
                par[y] = x
                time[y] = t_y + time[x]
                vis.add(y)
                heapq.heappush(q, (time[y] + heuristic[end][y]/sp, y))
    path = []
    tot_time = time[end]
    num_visited = len(vis)
    cur = end
    while cur != start:
        path.append(cur)
        cur = par[cur]
    path.append(start)
    path = list(reversed(path))
    return path, tot_time, num_visited
    # End your code (Part 6)


if __name__ == '__main__':
    path, time, num_visited = astar_time(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
