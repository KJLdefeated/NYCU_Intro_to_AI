import csv
import sys
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    sys.setrecursionlimit(20000)
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
    vis = set()
    dis = {}
    par = {}
    dis[start] = 0
    vis.add(start)
    def f(x):
        for y, dist in reversed(edge[x]):
            if y not in vis and y in edge:
                par[y] = x
                dis[y] = dist + dis[x]
                vis.add(y)
                f(y)
    f(start)
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
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
