import matplotlib
matplotlib.use('Qt5Agg')
# create directed graph

def populate_graph(graph, parent):
    for child in parent.children:
        graph.add_edge(parent.name, child.name)
        if getattr(child, 'children', None) is None:
            continue
        else:
            if len(child.children) > 0:
                populate_graph(graph, child)