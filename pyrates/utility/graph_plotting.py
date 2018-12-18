"""Collection of functions to plot a graph from a circuit/networkx graph."""

from pyrates.ir.circuit import CircuitIR


def simple_plot(circuit: CircuitIR, _format: str = "png", path: str = None, prog="dot", **pydot_args):
    """Simple straight plot using graphviz via pydot.

    Parameters
    ----------
    circuit
        `CircuitIR` instance to plot graph from
    _format
        output format
    path
        path to print image to. If `None` is given, will try to plot using matplotlib/IPython
    prog
        graphviz layout algorithm name. Defaults to "dot". Another recommendation is "circo". Other valid options:
        "fdp", "neato", "osage", "patchwork", "twopi", "pydot_args". See graphviz documentation for more info.

    Returns
    -------

    """

    import networkx as nx
    # import pydot

    # copy plain node names and plain edges, to avoid conflicts with graphviz
    graph = nx.MultiDiGraph()
    nodes = (node for node in circuit.graph.nodes)
    graph.add_nodes_from(nodes)
    edges = (edge for edge in circuit.graph.edges)
    graph.add_edges_from(edges)

    # pass graph to pydot
    dot_graph = nx.drawing.nx_pydot.to_pydot(graph)

    # check and format given path
    if path:
        return write_graph(path, dot_graph, prog, **pydot_args)

    else:
        return show_graph(dot_graph, _format, prog, **pydot_args)


def plot_graph_with_subgraphs(circuit: CircuitIR, _format: str = "png", path: str = None, prog="dot",
                              node_style="solid", cluster_style="rounded", **pydot_args):
    """Simple straight plot using graphviz via pydot.

    Parameters
    ----------
    circuit
        `CircuitIR` instance to plot graph from
    _format
        output format
    path
        path to print image to. If `None` is given, will try to plot using matplotlib/IPython
    prog
        graphviz layout algorithm name. Defaults to "dot". Another recommendation is "circo". Other valid options:
        "fdp", "neato", "osage", "patchwork", "twopi", "pydot_args". See graphviz documentation for more info.

    Returns
    -------

    """

    import pydot

    # copy plain node names and plain edges, to avoid conflicts with graphviz
    graph = pydot.Dot(graph_type='digraph', fontname="Verdana")
    clusters = {}
    for subcircuit in circuit.sub_circuits:
        clusters[subcircuit] = pydot.Cluster(subcircuit, label=subcircuit)

    for label, data in circuit.nodes(data=True):
        # node = data["node"]
        *subcircuit, node = label.split("/")
        node = pydot.Node(label, label=node.split(".")[0])
        node.set_style(node_style)

        if subcircuit:
            subcircuit = "/".join(subcircuit)
            cluster = clusters[subcircuit]
            cluster.add_node(node)
        else:
            graph.add_node(node)

    for cluster in clusters.values():
        cluster.set_style(cluster_style)
        graph.add_subgraph(cluster)

    for source, target, _ in circuit.edges:
        graph.add_edge(pydot.Edge(source, target))

    #
    # # pass graph to pydot
    # dot_graph = nx.drawing.nx_pydot.to_pydot(graph)
    #
    # # check and format given path
    if path:
        return write_graph(path, graph, prog, **pydot_args)

    else:
        return show_graph(graph, _format, prog, **pydot_args)


def write_graph(path, dot_graph, prog, **pydot_args):
    """Write DOT graph to file.

    Parameters
    ----------
    path
    dot_graph
    prog
    pydot_args

    Returns
    -------

    """
    import os
    path = os.path.normpath(path)

    # draw graph and write to file, given the format
    return dot_graph.write(path, format=format, prog=prog, **pydot_args)


def show_graph(dot_graph, _format, prog, **pydot_args):
    """Show DOT graph using matplotlib or IPython.display
    
    Parameters
    ----------
    dot_graph
    _format
    prog
    pydot_args

    Returns
    -------

    """

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    if _format == "png":
        # draw graph and write out postscript string
        image_bytes = dot_graph.create_png(prog=prog, **pydot_args)

        # check interactive session
        try:
            import sys
            _ = sys.ps1
        except AttributeError:
            # treat the dot output bytes as an image file
            from io import BytesIO
            bio = BytesIO()
            bio.write(image_bytes)
            bio.seek(0)
            img = mpimg.imread(bio)

            # plot the image
            imgplot = plt.imshow(img, aspect='equal')
            plt.show(block=False)
            return imgplot
        else:
            from IPython.display import Image, display

            return Image(data=image_bytes)

    else:
        raise NotImplementedError(f"No plotting option implemented for format '{_format}'")
