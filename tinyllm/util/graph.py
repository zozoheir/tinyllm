import networkx as nx
from matplotlib import pyplot as plt

from tinyllm.functions.chain import Chain
from tinyllm.functions.decision import Decision
from tinyllm.functions.concurrent import Concurrent


def graph_chain(chain):
    G = nx.DiGraph()
    populate_graph(G, chain)
    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]["layer"] = layer
    pos = nx.multipartite_layout(G, subset_key="layer")

    # Calculate new positions for the labels to add a bottom margin
    label_pos = {node: (coords[0], coords[1] + 0.02) for node, coords in pos.items()}  # Adjust as necessary

    fig, ax = plt.subplots()

    # Define colors and shapes for different node types
    colors = {
        "Decision": "#254bc1",
        "Chain": "#8d9bc4",
        "Concurrent": "#bcbcbc",
        "default": "#000000",
    }
    shapes = {
        "Decision": "s",  # square
        "Chain": "h",  # triangle
        "Concurrent": "o",  # circle
        "default": "^",  # hexagon
    }

    # Draw nodes with their respective colors and shapes
    for node in G.nodes:
        node_type = node.split(":")[0]
        color = colors.get(node_type, colors["default"])
        shape = shapes.get(node_type, shapes["default"])
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_shape=shape, ax=ax)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax)

    # Draw node labels with the adjusted positions
    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, label_pos, labels=node_labels, verticalalignment='bottom', ax=ax)

    ax.set_title(f"{chain.name} compute graph")
    fig.tight_layout()
    plt.show()


def get_node_name(node):
    if node.__class__ in [Decision, Chain, Concurrent]:
        return f"{node.__class__.__name__}: {node.name}"
    else:
        return node.name

def populate_graph(networkx_graph, function_node, parent_function_node=None):
    """
    Populates a NetworkX graph with edges representing function calls
    :param networkx_graph: The NetworkX graph to populate.
    :param function_node: The current function node being processed.
    :param parent_function_node: The parent function node of the current function node.
    """
    # Add an edge from the parent function node to the current function node
    if parent_function_node is not None:
        networkx_graph.add_edge(get_node_name(parent_function_node), get_node_name(function_node))

    # If the current function node is a Chain, add edges between its children
    if isinstance(function_node, Chain):
        for i in range(len(function_node.children) - 1):
            networkx_graph.add_edge(get_node_name(function_node.children[i]), get_node_name(function_node.children[i+1]))
            populate_graph(networkx_graph, function_node.children[i], function_node)

    # If the current function node is a Concurrent, add edges from it to all its children
    elif isinstance(function_node, Concurrent):
        for child in function_node.children:
            networkx_graph.add_edge(get_node_name(function_node), get_node_name(child))
            populate_graph(networkx_graph, child, function_node)

    # Recursively populate the graph with any nested functions
    if hasattr(function_node, "children"):
        for func in function_node.children:
            populate_graph(networkx_graph, func, function_node)