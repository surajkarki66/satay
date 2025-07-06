import sys
import onnx
import json
import numpy as np
import onnx_graphsurgeon as gs

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

# definition of the model name
input_model_path = sys.argv[1]
output_model_path = sys.argv[2]
input_shape = int(sys.argv[3])

# Load and import ONNX model
onnx_model = onnx.load(input_model_path)
graph = gs.import_onnx(onnx_model)

# Names of reshape (post-processing) nodes to remove
reshapes_to_remove = {
    "/model.22/Reshape",
    "/model.22/Reshape_1",
    "/model.22/Reshape_2"
}

# Remove reshape nodes by filtering
graph.nodes = [node for node in graph.nodes if node.name not in reshapes_to_remove]

# Get Concat nodes
concat_names = [
    "/model.22/Concat",
    "/model.22/Concat_1",
    "/model.22/Concat_2"
]

concat_nodes = {}
for name in concat_names:
    node = next((n for n in graph.nodes if n.name == name), None)
    if node is None:
        raise ValueError(f"Concat node {name} not found in graph.")
    concat_nodes[name] = node

# Update Resize nodes' ROI inputs
for resize_name, roi_name in [("/model.10/Resize", "roi_0"), ("/model.13/Resize", "roi_1")]:
    resize_node = next((n for n in graph.nodes if n.name == resize_name), None)
    if resize_node is None:
        raise ValueError(f"Resize node {resize_name} not found.")
    if len(resize_node.inputs) > 1:
        resize_node.inputs[1] = gs.Constant(roi_name, np.zeros(4, dtype=np.float16))

# Create new graph outputs based on concat outputs
graph.outputs = []
for name, node in concat_nodes.items():
    output_name = f"{name}_output_0"
    output_var = gs.Variable(output_name, shape=node.outputs[0].shape, dtype=np.float16)
    node.outputs = [output_var]
    graph.outputs.append(output_var)

# Clean up graph and export
graph.cleanup()
onnx_graph = gs.export_onnx(graph)
onnx_graph.ir_version = 8  # Downgrade IR version for compatibility
onnx.save(onnx_graph, output_model_path)
