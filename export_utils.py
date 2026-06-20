import onnx
import torch

def export_to_onnx(model, filename, input_shape=(1, 10, 30, 30)):
    """Exports a PyTorch model to ONNX with static shapes."""
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=18,  # Use a modern opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Model exported to {filename} (opset 18)")
