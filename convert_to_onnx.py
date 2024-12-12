import torch
import onnx
import os
from onnxsim import simplify
from model_base import SimpleCNN
from model_base import SimpleNet2D

def convert_to_onnx():
    print("Loading PyTorch model...")

    # Create output directory if it doesn't exist
    output_dir = "output_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Model Input
    model = SimpleCNN(7)
    
    # Load checkpoint
    checkpoint = torch.load('./runs/FurnitureClassification-41/best_checkpoint.pth', 
                          map_location=torch.device('cpu'))
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 177, 177)
    
    # Define output paths
    temp_model_path = os.path.join(output_dir, "temp_model.onnx")
    final_model_path = os.path.join(output_dir, "furniture_model.onnx")
    
    # First export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        temp_model_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Load and simplify
    print("Simplifying ONNX model...")
    onnx_model = onnx.load(temp_model_path)
    model_simp, check = simplify(onnx_model)
    
    if check:
        print("Simplified ONNX model was validated")
        onnx.save(model_simp, final_model_path)
        print(f"Saved simplified model to {final_model_path}")
        
        # Clean up temporary file
        os.remove(temp_model_path)
        print("Removed temporary model file")
    else:
        print("Simplified ONNX model could not be validated")
        return False

    print("Conversion complete!")
    return True

if __name__ == "__main__":
    convert_to_onnx()