import csv
import heapq
edgeFile = 'edges.csv'


def ucs(start, end):
    # Begin your code (Part 3)
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
    q = [(0, start)]
    vis = set()
    dis = {}
    par = {}
    dis[start] = 0
    vis.add(start)
    while q:
        dist, x = heapq.heappop(q)
        if x not in edge:
            continue
        for y, d_y in edge[x]:
            if y not in dis or dis[y] > d_y + dist:
                par[y] = x
                dis[y] = d_y + dist
                vis.add(y)
                heapq.heappush(q, (dis[y], y))
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
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
