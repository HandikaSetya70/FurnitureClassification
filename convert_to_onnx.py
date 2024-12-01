import torch
from model_base import SimpleCNN

def convert_to_onnx(model_path, output_path, input_shape=(1, 3, 177, 177)):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = SimpleCNN(7)  # 7 classes
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state' in state_dict:
        model.load_state_dict(state_dict['model_state'])
    else:
        model.load_state_dict(state_dict)
    
    # Move model to device and set to eval mode
    model = model.to(device)  # Add this line
    model.eval()
    
    # Create dummy input tensor - make sure it's on the same device
    dummy_input = torch.randn(input_shape).to(device)  # Modified this line
    
    # Move model back to CPU for ONNX export
    model = model.cpu()  # Add this line
    dummy_input = dummy_input.cpu()  # Add this line
    
    # Export to ONNX
    torch.onnx.export(
        model,                     # model being run
        dummy_input,              # model input (or a tuple for multiple inputs)
        output_path,              # where to save the model
        export_params=True,       # store the trained parameter weights inside the model file
        opset_version=11,         # the ONNX version to export the model to
        do_constant_folding=True, # whether to execute constant folding
        input_names=['input'],    # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},  # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model has been converted to ONNX and saved at {output_path}")

if __name__ == "__main__":
    # Path to your trained PyTorch model
    model_path = "./runs/FurnitureClassification-37/best_checkpoint.pth"
    
    # Path where you want to save the ONNX model
    output_path = "furniture_classifier.onnx"
    
    # Convert the model
    convert_to_onnx(model_path, output_path)
    
    # Verify the model
    import onnx
    
    # Load the ONNX model
    model = onnx.load(output_path)
    
    # Check that the model is well formed
    onnx.checker.check_model(model)
    
    print("Model was successfully exported and verified!")