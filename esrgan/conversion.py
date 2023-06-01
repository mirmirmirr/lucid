import torch
import torchvision
import onnx
import onnx2keras

# Step 1: Convert PyTorch model to ONNX format
torch_model = torchvision.models  # Load your PyTorch model here
torch_model.load_state_dict(torch.load('esrgan/model.pth')['params_ema'])

# weight = torchvision.models.resnet50(weights = 'esrgan/model.pth')
# torch_model = weight

dummy_input = torch.randn(1, 3, 224, 224)  # Provide a dummy input shape
onnx_path = 'model.onnx'

torch.onnx.export(torch_model, dummy_input, onnx_path, export_params=True)

# Step 2: Convert ONNX model to Keras model
onnx_model = onnx.load(onnx_path)
keras_model = onnx2keras.onnx_to_keras(onnx_model)

# Step 3: Load weights into Keras model
keras_model.load_weights('model.pth', by_name=True)

# Now you can use the Keras model for inference or further processing
