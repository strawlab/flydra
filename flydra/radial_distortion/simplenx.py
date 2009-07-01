import collections
import pprint

def sorted_keys( mydict ):
    """provides deterministic ordering of keys from dictionary"""
    keys = mydict.keys()
    keys.sort()
    return keys

class Graph:

    def __init__(self,vals=None):

        self._adjacency = collections.defaultdict(dict)
        self._nodes = []
        if vals is not None:
            for u,vs in vals.iteritems():
                for v in vs:
                    self.add_edge(u,v)

    def __repr__(self):
        # XXX could return a more sparse representation
        return "Graph(%s)"%repr(dict(self._adjacency))

    def add_edge(self,u,v=None):

        if v is None:
            u,v = u

        self._adjacency[u][v] = None
        self._adjacency[v][u] = None

        if u not in self._nodes:
            self._nodes.append( u )

        if v not in self._nodes:
            self._nodes.append( v )

    def neighbors(self, n):
        return sorted_keys(self._adjacency[n])

    def has_edge(self, u, v):
        return ((u in self._adjacency) and (v in self._adjacency[u]))

    def edges(self):
        edges = []
        for u in sorted_keys(self._adjacency):
            for v in self._adjacency[u]:
                if (v,u) not in edges:
                    edges.append( (u,v) )
        return edges

    def nodes(self):
        return [u for u in sorted_keys(self._adjacency)]

    def __eq__(self,other):
        return (self.edges() == other.edges() and
                self.nodes() == other.nodes())

def test_graph_repr():
    g = Graph()
    g.add_edge(1,2)
    g.add_edge(1,4)
    g.add_edge(2,8)
    g.add_edge(2,1)
    g.add_edge('x','y')
    gs = repr(g)
    g2 = eval(gs)
    assert g == g2

class Search:
    def dfs_preorder(self, G, source=None, reverse_graph=False):
        if source is None:
            nlist=G.nodes() # process entire graph
        else:
            nlist=[source]  # only process component with source

        if reverse_graph==True:
            try:
                neighbors=G.in_neighbors
            except:
                neighbors=G.neighbors
        else:
            neighbors=G.neighbors

        seen={} # nodes seen
        pre=[]  # list of nodes in a DFS preorder
        for source in nlist:
            if source in seen: continue
            queue=[source]     # use as LIFO queue
            while queue:
                v=queue[-1]
                if v not in seen:
                    pre.append(v)
                    seen[v]=True
                done=1
                for w in neighbors(v):
                    if w not in seen:
                        queue.append(w)
                        done=0
                        break
                if done==1:
                    queue.pop()
        return pre

search = Search()

def draw(G, pos=None, width=None):
    import pylab
    if pos is None:
        raise NotImplementedError('')

    for node in G.nodes():
        x,y = pos[node]
        pylab.plot( [x],[y], 'o' ,
                    markersize=20,
                    mec='b',
                    mfc='w',
                    zorder = 1,
                    )
        pylab.text( x,y, repr(node),
                    horizontalalignment='center',
                    verticalalignment='center',
                    zorder = 2,
                    color='b',
                    )

    for edge in G.edges():
        x0,y0 = pos[edge[0]]
        x1,y1 = pos[edge[1]]
        pylab.plot( [x0,x1],[y0,y1],
                    'b-',
                    lw=width,
                    zorder = 0,
                    )
