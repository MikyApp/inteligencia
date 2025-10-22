#1. Instalar lar librerias necesarias
#pip install roboflow ultralytics


#2.  Descargar y preparar el dataset desde Roboflow
#from roboflow import Roboflow
#
#rf = Roboflow(api_key="HKOSagd8b7b2VYzGgE98")
#project = rf.workspace("mik-jhpv0").project("deteccion_somnolencia")
#version = project.version(2)
#dataset = version.download("yolov5")  # Asegúrate de usar "yolov5" como formato

#3. Cargar el modelo base YOLOv5 paqueño preentrenado
#from ultralytics import YOLO
#model = YOLO("yolov5s.pt")  # modelo base YOLOv5 pequeño
#
#4. Entrenar el modelo personalizado con tu dataset
#import torch
#def main():
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    print("Entrenando en dispositivo:", device)
#
#    model = YOLO("yolov5su.pt")
#    model.model.to(device)  # Opcional, Ultralytics lo maneja
#
#    data_path = "deteccion_somnolencia-2/data.yaml"  # ruta al archivo yaml de tu dataset en formato YOLOv5
#    results = model.train(data=data_path,
#                          epochs=5,
#                          imgsz=480,  # tamaño recomendado para YOLOv5 (mayor resolución mejora precisión)
#                          batch=4,   # tamaño de batch recomendable según GPU
#                          workers=2)  # número de procesos para carga de datos
#
#if __name__ == "__main__":
#    import multiprocessing
#    multiprocessing.freeze_support()
#    multiprocessing.set_start_method('spawn', force=True)
#    main()
#

#5. Realizar predicciones con el mejor modelo entrenado
#custom_model = YOLO("runs/detect/train/weights/best.pt")
#results = custom_model("deteccion_somnolencia-2/test/images")  # ruta a las imágenes prueba
#
## Visualizar la primera imagen predicha
#results[0].show()
