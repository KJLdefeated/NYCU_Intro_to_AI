import csv
import heapq
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar(start, end):
    # Begin your code (Part 4)
    """
    First construc the edges from edge file then import the heuristic from heuristic csv.
    The method is like BFS, but we first consider the minimum heuristic value and distance from start.
    Record the distance of current point from start point and the parent of each node.
    Finally find the path.
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
            if st not in edge:
                edge[st] = []
            edge[st].append((ed, distance))
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
    dis = {}
    par = {}
    dis[start] = 0
    vis.add(start)
    while q:
        f, x = heapq.heappop(q)
        if x == end:
            break
        if x not in edge:
            continue
        for y, d_y in edge[x]:
            if y not in dis or dis[y] > d_y + dis[x]:
                par[y] = x
                dis[y] = d_y + dis[x]
                vis.add(y)
                heapq.heappush(q, (dis[y] + heuristic[end][y], y))
    path = []
    dist = dis[end]
    num_visited = len(vis)
    cur = end
    while cur != start:
        path.append(cur)
        cur = par[cur]
    path.append(start)
    path = list(reversed(path))
    return path, dist, num_visited
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
