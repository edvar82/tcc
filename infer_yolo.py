import onnxruntime
import cv2
import numpy as np

def test_yolov8n_onnx(model_path, image_path, class_names):
    # Carregar o modelo ONNX
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Carregar a imagem
    image = cv2.imread(image_path)
    original_height, original_width, _ = image.shape

    # Pré-processar a imagem
    resized_image = cv2.resize(image, (640, 640))
    input_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0

    # Realizar a inferência
    outputs = session.run(None, {input_name: input_image})
    detections = outputs[0][0]  # Obter as detecções

    # Pós-processar as detecções
    for detection in detections.T:  # Transpor para iterar sobre as detecções
        x1, y1, x2, y2 = detection[:4]  # Coordenadas da caixa delimitadora
        confidence = detection[4]  # Confiança da detecção
        class_probs = detection[5:]  # Probabilidades das classes
        class_id = np.argmax(class_probs)  # ID da classe

        if confidence > 0.5:  # Limiar de confiança
            x1, y1, x2, y2 = map(int, [x1 * original_width, y1 * original_height, x2 * original_width, y2 * original_height])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{class_names[class_id]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibir a imagem com as detecções
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Caminhos e nomes das classes
model_path = "models/yolov8n.onnx"  # Substitua pelo caminho do seu modelo .onnx
image_path = "teste.jpg"  # Substitua pelo caminho da sua imagem
class_names = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

# Executar o teste
test_yolov8n_onnx(model_path, image_path, class_names)

# import onnxruntime
# import numpy as np

# model_path = "models/best.onnx" # Coloque aqui o caminho do seu arquivo .onnx
# session = onnxruntime.InferenceSession(model_path)
# input_name = session.get_inputs()[0].name

# # Criar uma entrada fictícia para obter a forma da saída
# input_shape = session.get_inputs()[0].shape
# dummy_input = np.random.rand(*input_shape).astype(np.float32)

# outputs = session.run(None, {input_name: dummy_input})

# print("Forma da saída:", outputs[0].shape)
# print("Exemplo de detecções:", outputs[0][0][:5]) # Imprime as 5 primeiras detecções

# from ultralytics import YOLO
# import cv2

# def test_yolov8n_pt(model_path, image_path):
#     # Carregar o modelo YOLOv8n .pt
#     model = YOLO(model_path)

#     # Realizar a inferência na imagem
#     results = model(image_path)

#     # Processar os resultados
#     for result in results:
#         boxes = result.boxes.cpu().numpy()  # Obter as caixas delimitadoras
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da caixa
#             class_id = int(box.cls[0])  # ID da classe
#             confidence = box.conf[0]  # Confiança da detecção
#             print(f"{model.names[class_id]} {confidence:.2f}")

#             # Desenhar a caixa e o rótulo na imagem
#             image = cv2.imread(image_path)
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, f"{model.names[class_id]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Exibir a imagem com as detecções
#         cv2.imshow("Detections", image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# # Caminho para o modelo .pt e a imagem de teste
# model_path = "models/best.pt"  # Substitua pelo caminho do seu modelo
# image_path = "teste.jpg"  # Substitua pelo caminho da sua imagem

# # Executar o teste
# test_yolov8n_pt(model_path, image_path)