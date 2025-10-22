# 1. Instalara librerias
# pip install roboflow ultralytics

# 2. Descargar el dataset de roboflow
# from roboflow import Roboflow

# rf = Roboflow(api_key="HKOSagd8b7b2VYzGgE98")
# project = rf.workspace("mik-jhpv0").project("deteccion_somnolencia")
# version = project.version(2)
# dataset = version.download("yolov11")

# 3. Cargar el modelo base YOLOv11
# from ultralytics import YOLO
# model = YOLO("yolo11s.pt")

# 4. Entrenamiento del modelo personalizado
#from ultralytics import YOLO
#import torch
#
#
#def main():
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    print("Entrenando en dispositivo:", device)
#
#    model = YOLO("yolo11s.pt")
#    # Opcional: mover modelo a dispositivo explícitamente (aunque Ultralytics lo maneja)
#    model.model.to(device)
#
#    data_path = "deteccion_somnolencia-2/data.yaml"
#    results = model.train(data=data_path,
#                          epochs=5,
#                          imgsz=360,
#                          batch=4,
#                          workers=0)  # pasar dispositivo al train para asegurar
#
#if __name__ == "__main__":
#    import multiprocessing
#    multiprocessing.freeze_support()
#    multiprocessing.set_start_method('spawn', force=True)
#    main()

# 5. Hacer predicciones
#Cargamos el mejor modelo entrenado
#custom_model = YOLO("runs/detect/train/weights/best.pt")

#Realizamos predicciones sobre algunas imágenes
#res = custom_model("deteccion_somnolencia-2/test/images")

#Visualizar en images
#res[400].show()
