import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    """
    Read csv file and construct edges.
    Using stack to implement dfs. Stop searching util find the end point.
    Record the distance of current point from start point and the parent of each node.
    Finally find the path.
    """
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
    stack = [start]
    vis = set()
    dis = {}
    par = {}
    dis[start] = 0
    vis.add(start)
    while stack:
        x = stack.pop()
        if x not in edge:
            continue
        for y, dist in edge[x]:
            if y not in vis:
                par[y] = x
                dis[y] = dist + dis[x]
                vis.add(y)
                stack.append(y)
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
