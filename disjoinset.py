class DisjoinSet(object):
    def __init__(self, n):
        self.n = n
        self.par = [i for i in range(self.n)]
        self.count = self.n
        self.cluster_label = []
        self.correspond_root = []

    def find(self, x):
        root = x
        while self.par[root] != root:
            root = self.par[root]
        while x != root:
            tmp = self.par[x]
            self.par[x] = root
            x = tmp
        return root

    def merge(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.par[px] = py
            self.count -= 1

    def is_same(self, x, y):
        return self.find(x) == self.find(y)

    def arrange(self):
        no = [-1 for _ in range(self.n)]
        cnt = 0
        for i in range(self.n):
            p = self.find(i)
            if no[p] == -1:
                no[p] = cnt
                self.correspond_root.append(p)
                cnt += 1
            self.cluster_label.append(no[p])


if __name__ == "__main__":
    n = int(input())
    uf = DisjoinSet(n)
    while True:
        x, y = list(map(int, input().split()))
        if x == -1:
            break
        uf.merge(x, y)
        print([uf.find(i) for i in range(n)])
    uf.arrange()
    print("Cluster: ", uf.cluster_label)
    print("Root: ", uf.correspond_root)
