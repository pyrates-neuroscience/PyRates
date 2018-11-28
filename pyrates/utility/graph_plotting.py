"""Collection of functions to plot a graph from a circuit/networkx graph."""

from pyrates.ir.circuit import CircuitIR


def simple_plot(circuit: CircuitIR, format: str="png", path: str=None, **pydot_args):
    """Simple straight plot"""

    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import pydot

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
        import os
        path = os.path.normpath(path)

        # draw graph and write to file, given the format
        return dot_graph.write(path, format=format, **pydot_args)

    else:

        if format == "png":
            # draw graph and write out postscript string
            image_bytes = dot_graph.create_png(**pydot_args)

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
            raise NotImplementedError(f"No plotting option implemented for format '{format}'")
