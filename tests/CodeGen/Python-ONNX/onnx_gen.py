import onnx
from onnx import helper, TensorProto
import random
import numpy as np
import argparse
import datetime
from onnx import StringStringEntryProto

def random_shape(rank, min_dim=1, max_dim=10):
    """Generates a random shape of length 'rank'."""
    return [random.randint(min_dim, max_dim) for _ in range(rank)]

def generate_fuzz_model(op_name):
    """
    Creates inputs, outputs, nodes and any initializers with random parameters 
    for the given op_name.
    """
    initializers = []
    
    # Generate descriptive names for inputs and outputs
    input_names = [f"{op_name}_input{i}" for i in range(5)]  # Pre-generate input names
    output_names = [f"{op_name}_output{i}" for i in range(5)]  # Pre-generate output names
    metadata = {}  # Store metadata about the model
    
    if op_name in ["Relu", "Sigmoid", "Ceil", "Tanh", "Identity", "Neg", "Shape"]:
        # Single-input operators with a random shape (rank=4)
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                               name=f"{op_name}_node")
        metadata = {"input_shapes": [shape], "output_shapes": [shape]}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "LeakyRelu":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        alpha = round(random.uniform(0.001, 0.2), 3)
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                               alpha=alpha, name=f"{op_name}_node_alpha_{alpha}")
        metadata = {"input_shapes": [shape], "output_shapes": [shape], "alpha": alpha}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Softmax":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(-rank, rank-1)
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                               axis=axis, name=f"{op_name}_node_axis_{axis}")
        metadata = {"input_shapes": [shape], "output_shapes": [shape], "axis": axis}
        return input_info, output_info, [node], initializers, metadata

    elif op_name in ["Add", "Sub", "Mul", "Div", "Mean"]:
        # Binary operators: two inputs with the same shape
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        input_info = [
            helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape),
            helper.make_tensor_value_info(input_names[1], TensorProto.FLOAT, shape)
        ]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]], 
                               name=f"{op_name}_node")
        metadata = {"input_shapes": [shape, shape], "output_shapes": [shape]}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Concat":
        # Two inputs with identical shape except along the concatenation axis
        shape = [1, random.randint(2,5), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(0, rank-1)
        shape2 = shape.copy()
        shape2[axis] = shape[axis] + random.randint(1,3)
        input_info = [
            helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape),
            helper.make_tensor_value_info(input_names[1], TensorProto.FLOAT, shape2)
        ]
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] + shape2[axis]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]], 
                               axis=axis, name=f"{op_name}_node_axis_{axis}")
        metadata = {"input_shapes": [shape, shape2], "output_shapes": [out_shape]}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Gather":
        # First input: data, second input: indices (provided as an initializer to ensure valid values)
        shape = [5, random.randint(5,10)]
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        indices_shape = [random.randint(1,3)]
        input_info.append(helper.make_tensor_value_info(input_names[1], TensorProto.INT64, indices_shape))
        axis = random.randint(0, len(shape)-1)
        indices_data = np.random.randint(0, shape[axis], size=indices_shape).astype(np.int64)
        initializer = helper.make_tensor(input_names[1], TensorProto.INT64, indices_shape, indices_data.flatten().tolist())
        initializers.append(initializer)
        out_shape = list(shape)
        out_shape[axis] = indices_shape[0]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]], 
                               axis=axis, name=f"{op_name}_node_axis_{axis}")
        metadata = {"input_shapes": [shape, indices_shape], "output_shapes": [out_shape]}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Pad":
        # Generate random pads for each dimension
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        pads = [random.randint(0,2) for _ in range(2*rank)]
        out_shape = [shape[i] + pads[i] + pads[i+rank] for i in range(rank)]
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        
        # Create pads as a second input tensor (required in newer ONNX versions)
        pads_tensor = helper.make_tensor('pads', TensorProto.INT64, [len(pads)], pads)
        initializers.append(pads_tensor)
        input_info.append(helper.make_tensor_value_info(input_names[1], TensorProto.INT64, [len(pads)]))
        
        # Optional constant_value input (using 0.0 as default)
        constant_value = 0.0
        constant_tensor = helper.make_tensor('constant_value', TensorProto.FLOAT, [], [constant_value])
        initializers.append(constant_tensor)
        input_info.append(helper.make_tensor_value_info(input_names[2], TensorProto.FLOAT, []))
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]], outputs=[output_names[0]], 
                               name=f"{op_name}_node")
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "pads": pads, "constant_value": constant_value}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Reshape":
        # The second input is an initializer that contains the new shape (e.g., a permutation)
        shape = [random.randint(1,4) for _ in range(4)]
        total_elems = int(np.prod(shape))
        new_shape = shape.copy()
        random.shuffle(new_shape)
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        shape_tensor = helper.make_tensor('shape', TensorProto.INT64, [len(new_shape)], new_shape)
        initializers.append(shape_tensor)
        input_info.append(helper.make_tensor_value_info(input_names[1], TensorProto.INT64, [len(new_shape)]))
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, new_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]], 
                               name=f"{op_name}_node")
        metadata = {"input_shapes": [shape, new_shape], "output_shapes": [new_shape]}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Resize":
        # Four inputs: X, roi, scales, sizes
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        roi = []
        roi_tensor = helper.make_tensor('roi', TensorProto.FLOAT, [0], roi)
        initializers.append(roi_tensor)
        input_info.append(helper.make_tensor_value_info(input_names[1], TensorProto.FLOAT, [0]))
        scales = [round(random.uniform(0.5, 2.0), 2) for _ in shape]
        scales_tensor = helper.make_tensor('scales', TensorProto.FLOAT, [len(scales)], scales)
        initializers.append(scales_tensor)
        input_info.append(helper.make_tensor_value_info(input_names[2], TensorProto.FLOAT, [len(scales)]))
        sizes = [int(round(s * dim)) for s, dim in zip(scales, shape)]
        sizes_tensor = helper.make_tensor('sizes', TensorProto.INT64, [len(sizes)], sizes)
        initializers.append(sizes_tensor)
        input_info.append(helper.make_tensor_value_info(input_names[3], TensorProto.INT64, [len(sizes)]))
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, sizes)
        mode = random.choice(["nearest", "linear"])
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2], input_names[3]], outputs=[output_names[0]], 
                               mode=mode, name=f"{op_name}_node_mode_{mode}")
        roi_shape = [0]  # Empty shape for roi
        scales_shape = [len(scales)]
        sizes_shape = [len(sizes)]
        metadata = {"input_shapes": [shape], "output_shapes": [sizes], "mode": mode, "scales": scales, "sizes": sizes}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Slice":
        # Create random slice indices for each dimension
        shape = [random.randint(5,10) for _ in range(4)]
        rank = len(shape)
        starts, ends = [], []
        for d in shape:
            start = random.randint(0, d-1)
            end = random.randint(start+1, d)
            starts.append(start)
            ends.append(end)
        
        # Create input tensors for starts and ends
        starts_tensor = helper.make_tensor('starts', TensorProto.INT64, [len(starts)], starts)
        ends_tensor = helper.make_tensor('ends', TensorProto.INT64, [len(ends)], ends)
        initializers.append(starts_tensor)
        initializers.append(ends_tensor)
        
        out_shape = [ends[i] - starts[i] for i in range(rank)]
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        input_info.append(helper.make_tensor_value_info(input_names[1], TensorProto.INT64, [len(starts)]))
        input_info.append(helper.make_tensor_value_info(input_names[2], TensorProto.INT64, [len(ends)]))
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]], outputs=[output_names[0]], 
                               name=f"{op_name}_node")
        starts_shape = [len(starts)]
        ends_shape = [len(ends)]
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "starts": starts, "ends": ends}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Split":
        # Split into 2 parts along a random axis
        shape = [1, random.randint(4,10), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(0, rank-1)
        if shape[axis] < 2:
            shape[axis] = 2
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] // 2
        output_info = [
            helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape),
            helper.make_tensor_value_info(output_names[1], TensorProto.FLOAT, out_shape)
        ]
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0], output_names[1]], 
                               axis=axis, name=f"{op_name}_node_axis_{axis}")
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape, out_shape]}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Transpose":
        # Generate a random permutation for Transpose
        shape = [random.randint(1,4) for _ in range(4)]
        rank = len(shape)
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        perm = list(range(rank))
        random.shuffle(perm)
        out_shape = [shape[i] for i in perm]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                               perm=perm, name=f"{op_name}_node_perm_{perm}")
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "perm": perm}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Unsqueeze":
        # Insert a new dimension at a random axis
        shape = [random.randint(1,4) for _ in range(4)]
        rank = len(shape)
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        
        # Create axes as a second input tensor
        axis = random.randint(0, rank)
        axes = [axis]
        axes_tensor = helper.make_tensor('axes', TensorProto.INT64, [len(axes)], axes)
        initializers.append(axes_tensor)
        input_info.append(helper.make_tensor_value_info(input_names[1], TensorProto.INT64, [len(axes)]))
        
        out_shape = shape.copy()
        out_shape.insert(axis, 1)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]], 
                               name=f"{op_name}_node_axis_{axis}")
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "axes": axes}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Conv":
        # For Conv, generate an input shape [N, C, H, W] with H and W sufficiently large
        N = 1
        C = random.randint(1,4)
        H = random.randint(10,50)
        W = random.randint(10,50)
        input_shape = [N, C, H, W]
        kH = random.randint(2, max(2, H//2))
        kW = random.randint(2, max(2, W//2))
        kernel_shape = [kH, kW]
        M = random.randint(1,4)
        weight_shape = [M, C, kH, kW]
        input_info = [
            helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info(input_names[1], TensorProto.FLOAT, weight_shape)
        ]
        H_out = H - kH + 1
        W_out = W - kW + 1
        output_shape = [N, M, H_out, W_out]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]], 
                               kernel_shape=kernel_shape, name=f"{op_name}_node_kernel_{kernel_shape}")
        metadata = {"input_shapes": [input_shape, weight_shape], "output_shapes": [output_shape], "kernel_shape": kernel_shape}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "MatMul":
        # Generate two compatible 2D matrices
        M = random.randint(2,10)
        K = random.randint(2,10)
        N = random.randint(2,10)
        A_shape = [M, K]
        B_shape = [K, N]
        input_info = [
            helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, A_shape),
            helper.make_tensor_value_info(input_names[1], TensorProto.FLOAT, B_shape)
        ]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, [M, N])
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]], 
                               name=f"{op_name}_node")
        output_shape = [M, N]
        metadata = {"input_shapes": [A_shape, B_shape], "output_shapes": [output_shape]}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "Gemm":
        # Performs A * B + C with compatible shapes
        M = random.randint(2,10)
        K = random.randint(2,10)
        N = random.randint(2,10)
        A_shape = [M, K]
        B_shape = [K, N]
        C_shape = [M, N]
        input_info = [
            helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, A_shape),
            helper.make_tensor_value_info(input_names[1], TensorProto.FLOAT, B_shape),
            helper.make_tensor_value_info(input_names[2], TensorProto.FLOAT, C_shape)
        ]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, [M, N])
        alpha = round(random.uniform(0.5, 2.0), 2)
        beta = round(random.uniform(0.5, 2.0), 2)
        transA = random.choice([0, 1])
        transB = random.choice([0, 1])
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]], outputs=[output_names[0]], 
                               alpha=alpha, beta=beta, transA=transA, transB=transB, name=f"{op_name}_node_alpha_{alpha}_beta_{beta}_transA_{transA}_transB_{transB}")
        output_shape = [M, N]
        metadata = {"input_shapes": [A_shape, B_shape, C_shape], "output_shapes": [output_shape], "alpha": alpha, "beta": beta, "transA": transA, "transB": transB}
        return input_info, output_info, [node], initializers, metadata

    elif op_name == "MaxPool":
        # Generate a pooling layer with random kernel and stride
        N = 1
        C = random.randint(1,4)
        H = random.randint(10,50)
        W = random.randint(10,50)
        input_shape = [N, C, H, W]
        kernel_size = random.randint(2, max(2, min(H, W)//2))
        kernel_shape = [kernel_size, kernel_size]
        strides = [random.randint(1, kernel_size), random.randint(1, kernel_size)]
        H_out = (H - kernel_size) // strides[0] + 1
        W_out = (W - kernel_size) // strides[1] + 1
        output_shape = [N, C, H_out, W_out]
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, input_shape)]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                               kernel_shape=kernel_shape, strides=strides, name=f"{op_name}_node_kernel_{kernel_shape}_strides_{strides}")
        metadata = {"input_shapes": [input_shape], "output_shapes": [output_shape], "kernel_shape": kernel_shape, "strides": strides}
        return input_info, output_info, [node], initializers, metadata

    else:
        # Fallback for any operators not explicitly handled
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        input_info = [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                               name=f"{op_name}_generic_node")
        metadata = {"input_shapes": [shape], "output_shapes": [shape]}
        return input_info, output_info, [node], initializers, metadata

def generate_model(op_name, filename, model_id=0):
    input_info, output_info, nodes, initializers, metadata = generate_fuzz_model(op_name)
    
    # If there are multiple outputs, pass them as a list
    graph_outputs = output_info if isinstance(output_info, list) else [output_info]
    
    # Add doc strings to nodes for better documentation
    for node in nodes:
        node.doc_string = f"Test node for {op_name} operation with ID {model_id}"
    
    # Create the graph with a descriptive name
    graph = helper.make_graph(
        nodes,
        name=f"{op_name}_test_graph_{model_id}",
        inputs=input_info,
        outputs=graph_outputs,
        initializer=initializers,
        doc_string=f"Test graph for {op_name} operation with configuration: {metadata}"
    )
    
    # Add model metadata
    opset_imports = [helper.make_opsetid("", 13)]  # Use ONNX opset 13
    
    # Create model with metadata
    model = helper.make_model(
        graph, 
        producer_name='zant_test_generator',
        producer_version='1.0',
        domain='ai.zant.test',
        model_version=model_id,
        doc_string=f"Test model for {op_name} operation. Generated on {datetime.datetime.now().isoformat()}",
        opset_imports=opset_imports
    )
    
    # Add additional metadata as properties - using the correct method
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_metadata"
    meta_prop.value = str(metadata)
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_op"
    meta_prop.value = op_name
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_id"
    meta_prop.value = str(model_id)
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "generator_version"
    meta_prop.value = "1.0"
    model.metadata_props.append(meta_prop)
    
    # Check and save the model
    onnx.checker.check_model(model)
    onnx.save(model, filename)
    print(f"Fuzzed model for {op_name} (ID: {model_id}) saved to: {filename}")
    return metadata

def load_supported_ops(filename="available_operations.txt"):
    try:
        with open(filename, "r") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default operations.")
        return []

def main():
    parser = argparse.ArgumentParser(description="Generate fuzzed ONNX models for CI/CD.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of models to generate for each operation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random generation (for reproducibility).")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save generated models.")
    parser.add_argument("--metadata-file", type=str, default=None,
                        help="File to save metadata about generated models.")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    output_dir = args.output_dir
    if output_dir and not output_dir.endswith('/'):
        output_dir += '/'
        
    supported_ops = load_supported_ops() #TODO : Softmax has errors in parsing, it has been removed from available_operations.txt
    if not supported_ops:
        supported_ops = [  # Fallback default operations
            "LeakyRelu", "Relu", "Sigmoid", "Softmax", "Add", "Ceil", "Div", "Mul", "Sub", "Tanh",
            "Concat", "Gather", "Identity", "Neg", "Reshape", "Resize", "Shape", "Slice", 
            "Split", "Transpose", "Unsqueeze", "Mean", "Conv", "MatMul", "Gemm", "MaxPool"
        ]
    
    all_metadata = {}
    for op in supported_ops:
        op_metadata = []
        for i in range(args.iterations):
            # Make sure the filename is correctly formatted
            filename = f"{output_dir}{op}_{i}.onnx"
            try:
                metadata = generate_model(op, filename, i)
                op_metadata.append(metadata)
                print(f"Successfully generated model for {op} (ID: {i})")
            except Exception as e:
                print(f"Error generating model for {op} (ID: {i}): {e}")
        all_metadata[op] = op_metadata
    
    # Save metadata if requested
    if args.metadata_file:
        import json
        with open(args.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"Metadata saved to {args.metadata_file}")

if __name__ == "__main__":
    main()


# LeakyRelu
# Relu
# Sigmoid
# Add
# Ceil
# Div
# Mul
# Sub
# Tanh
# Concat
# Gather
# Identity
# Neg
# Reshape
# Resize
# Shape
# Slice
# Split
# Transpose
# Unsqueeze
# Mean
# Conv
# MatMul
# Gemm
# MaxPool