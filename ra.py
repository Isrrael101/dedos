import cv2
import numpy as np
from cv2 import aruco
import sys
import os
import time
from PIL import Image, ImageSequence

class RealidadAumentada:
    def __init__(self, camera_url):
        print(f"Conectando a la cámara IP en {camera_url}...")
        self.cap = cv2.VideoCapture(camera_url)
        
        if not self.cap.isOpened():
            print("Error: No se pudo conectar a la cámara IP")
            print("Verifica:")
            print("1. Que la URL es correcta")
            print("2. Que el teléfono y la computadora están en la misma red")
            print("3. Que la app IP Webcam está ejecutándose")
            sys.exit(1)
        else:
            print("Conexión exitosa a la cámara IP")
        
        # Configurar ArUco con parámetros optimizados
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        # Ajustar parámetros para mejor detección
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.minMarkerPerimeterRate = 0.03
        self.parameters.maxMarkerPerimeterRate = 4.0
        
        # Variables para el seguimiento
        self.last_corners = None
        self.last_detection_time = 0
        self.detection_timeout = 0.5  # segundos
        
        # Variables para el GIF
        self.gif_frames = []
        self.current_frame_index = 0
        self.last_frame_time = 0
        self.frame_delay = 0.1  # Tiempo entre frames del GIF (ajustable)
        
        # Cargar el GIF
        self.cargar_gif()
        
        # Generar y guardar el marcador de prueba
        self.generar_marcador()

    def __del__(self):
        """Destructor para liberar recursos"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def cargar_gif(self):
        """Carga el GIF y lo prepara para la superposición"""
        try:
            gif_path = 'mario-icegif-6.gif'  # Asegúrate de que tu GIF se llame así
            if not os.path.exists(gif_path):
                print(f"Error: No se encuentra el archivo {gif_path}")
                sys.exit(1)
            
            # Abrir el GIF
            gif = Image.open(gif_path)
            
            # Redimensionar si es muy grande
            max_size = 200
            width, height = gif.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                new_size = (new_width, new_height)
            else:
                new_size = (width, height)
            
            # Extraer y procesar cada frame
            for frame in ImageSequence.Iterator(gif):
                # Redimensionar frame
                frame = frame.resize(new_size, Image.Resampling.LANCZOS)
                # Convertir a formato OpenCV
                cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                self.gif_frames.append(cv_frame)
            
            if len(self.gif_frames) == 0:
                print("Error: No se pudieron extraer frames del GIF")
                sys.exit(1)
                
            print(f"GIF {gif_path} cargado exitosamente con {len(self.gif_frames)} frames")
            
        except Exception as e:
            print(f"Error al cargar el GIF: {e}")
            sys.exit(1)

    def generar_marcador(self):
        """Genera y guarda un marcador ArUco de prueba"""
        marker = aruco.generateImageMarker(self.aruco_dict, 0, 400)
        marker_with_border = cv2.copyMakeBorder(marker, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite("marcador_test.png", marker_with_border)
        print("\n¡IMPORTANTE! Se ha generado 'marcador_test.png'")
        print("Por favor:")
        print("1. Abre el archivo 'marcador_test.png'")
        print("2. Muestra este marcador a la cámara del teléfono")
    
    def obtener_siguiente_frame_gif(self):
        """Obtiene el siguiente frame del GIF considerando el tiempo transcurrido"""
        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_delay:
            self.current_frame_index = (self.current_frame_index + 1) % len(self.gif_frames)
            self.last_frame_time = current_time
        return self.gif_frames[self.current_frame_index]
    
    def suavizar_esquinas(self, nuevas_esquinas):
        """Suaviza la transición entre las esquinas detectadas"""
        if self.last_corners is None:
            self.last_corners = nuevas_esquinas
            return nuevas_esquinas
        
        alpha = 0.5
        esquinas_suavizadas = self.last_corners * (1 - alpha) + nuevas_esquinas * alpha
        self.last_corners = esquinas_suavizadas
        return esquinas_suavizadas
        
    def superponer_imagen(self, frame, corners, marker_id):
        try:
            marker_corners = corners[0]
            marker_corners = self.suavizar_esquinas(marker_corners)
            
            # Obtener el siguiente frame del GIF
            overlay_image = self.obtener_siguiente_frame_gif()
            h, w = overlay_image.shape[:2]
            pts_dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            
            homography, _ = cv2.findHomography(pts_dst, marker_corners)
            warped_image = cv2.warpPerspective(overlay_image, homography, (frame.shape[1], frame.shape[0]))
            
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, marker_corners.astype(np.int32), 255)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            frame = frame * (1 - mask_3channel) + warped_image * mask_3channel
            
            return frame.astype(np.uint8)
            
        except Exception as e:
            print(f"Error al superponer imagen: {e}")
            return frame
    
    def ejecutar(self):
        print("\nIniciando realidad aumentada con GIF...")
        print("Presiona 'q' para salir")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error al leer frame de la cámara IP")
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    print(f"FPS: {fps:.2f}")
                
                height, width = frame.shape[:2]
                if width > 1000:
                    scale = 1000 / width
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
                
                current_time = time.time()
                
                if ids is not None or (current_time - self.last_detection_time) < self.detection_timeout:
                    if ids is not None:
                        self.last_detection_time = current_time
                        aruco.drawDetectedMarkers(frame, corners, ids)
                        
                        for i in range(len(ids)):
                            frame = self.superponer_imagen(frame, [corners[i]], ids[i])
                    elif self.last_corners is not None:
                        frame = self.superponer_imagen(frame, [self.last_corners], None)
                
                cv2.imshow('Realidad Aumentada con GIF', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error en el bucle principal: {e}")
        
        print("Cerrando aplicación...")
        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    url = "http://192.168.1.7:8080/video"
    ar = None
    
    try:
        ar = RealidadAumentada(url)
        ar.ejecutar()
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
    except Exception as e:
        print(f"Error inesperado: {e}")
    finally:
        if ar is not None:
            ar.cap.release()
        # Forzar el cierre de todas las ventanas
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Este waitKey adicional ayuda a forzar el cierre
        time.sleep(0.5)  # Pequeña pausa para asegurar el cierre
        cv2.destroyAllWindows()  # Llamada adicional para asegurar