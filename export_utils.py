import onnx
import torch

def export_to_onnx(model, filename, input_shape=(1, 10, 30, 30)):
    """Exports a PyTorch model to ONNX with static shapes and competition Opset 10."""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=10,  # Competition Opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    # Ensure shape inference for value_info
    m = onnx.load(filename)
    m = onnx.shape_inference.infer_shapes(m)
    onnx.save(m, filename)
    print(f"Model exported to {filename} (opset 10 with shape info)")
