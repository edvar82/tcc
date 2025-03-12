from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO("yolov8n.pt")  # Carrega o modelo pré-treinado
    model.train(data="Dataset/Pascal_VOC/data.yaml", epochs=50)  # Treina o modelo
    

if __name__ == '__main__':
    freeze_support()
    main()