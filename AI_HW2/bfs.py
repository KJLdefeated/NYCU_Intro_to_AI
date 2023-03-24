import csv
from collections import deque
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
    edge = {}
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
    q = deque([start])
    vis = set()
    dis = {}
    par = {}
    dis[start] = 0
    vis.add(start)
    while q:
        x = q.popleft()
        if x == end:
            break
        if x not in edge:
            continue
        for y, dist in edge[x]:
            if y not in vis:
                par[y] = x
                dis[y] = dist + dis[x]
                vis.add(y)
                q.extend([y])
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
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
