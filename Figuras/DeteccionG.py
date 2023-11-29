import numpy as np
import cv2
import os

# Rutas donde se guardarán las imágenes
ruta_guardado = r"M:\9no Semestre\WONG\Figuras"
ruta_guardado_fotos = os.path.join(ruta_guardado, "Guardadas")

# Crear la carpeta para las fotos guardadas si no existe
if not os.path.exists(ruta_guardado_fotos):
    os.makedirs(ruta_guardado_fotos)

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX

num_figuras = 1  # Inicializar el número de figuras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 180)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        vertices = len(approx)
        if vertices == 3:
            shape = "Figura desconocida"
        elif vertices == 4:
            shape = "Figura desconocida"
        elif vertices > 4:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.75:
                shape = "Figura desconocida"
            else:
                shape = "Figura desconocida"
        else:
            shape = "Figura desconocida"

        M = cv2.moments(contour)
        if M['m00'] != 0:  # que no sea 0
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Mostrar la etiqueta "Figura desconocida" y número de figura en la imagen
            cv2.putText(frame, f"{shape} {num_figuras}", (cx - 20, cy), font, 0.5, (0, 0, 0), 2)
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

            # Mostrar la imagen antes de guardarla
            cv2.imshow("Imagen a guardar", frame)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('s'):  # Si se presiona 's', guardar la imagen
                shape_name = input(f"Ingrese el nombre para la figura {num_figuras}: ")
                filename = os.path.join(ruta_guardado_fotos, f"{shape_name}_{num_figuras}.jpg")
                cv2.imwrite(filename, frame)
                num_figuras += 1  # Incrementar el número de figuras
            
            elif key == ord('4'):  # Si se presiona '4', cerrar la cámara y terminar el programa
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("Figura Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
