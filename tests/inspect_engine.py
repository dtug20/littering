import tensorrt as trt

def inspect_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        
        print(f"Engine: {engine_path}")
        print(f"Number of bindings: {engine.num_io_tensors}")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            print(f"Tensor {i}: Name='{name}', Input={is_input}, Shape={shape}, Dtype={dtype}")

inspect_engine("models/yolo11s.engine")
