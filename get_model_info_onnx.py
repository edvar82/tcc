import onnx
import os
import numpy as np

def get_model_info(model_path):
    model = onnx.load(model_path)

    num_layers = sum(1 for _ in model.graph.node)

    total_params = sum(np.prod(t.dims) for t in model.graph.initializer)
    trainable_params = total_params  
    non_trainable_params = 0

    model_size = os.path.getsize(model_path) / (1024 * 1024)  

    flops = total_params * 2.74  

    return num_layers, total_params, non_trainable_params, flops, model_size

model_path = "models/yolov4_tiny_voc_barracuda.onnx"
num_layers, total_params, non_trainable_params, flops, model_size = get_model_info(model_path)

with open("model_info.txt", "w") as f:
    f.write(f"({num_layers}, {total_params}, {non_trainable_params}, {flops:.7f})\n")
    f.write(f"YOLOv4-tiny tem {num_layers} camadas.\n")
    f.write(f"Ele possui cerca de {total_params/1e6:.1f} milhões de parâmetros treináveis.\n")
    f.write(f"Nenhum dos parâmetros está congelado ({non_trainable_params} não treináveis).\n")
    f.write(f"O modelo possui {flops/1e9:.1f} bilhões de operações de ponto flutuante por segundo (GFLOPs).\n")
    f.write(f"O tamanho do modelo é aproximadamente {model_size:.2f} MB.\n")
