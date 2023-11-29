
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX

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
            shape = "Triangulo"
        elif vertices == 4:
            shape = "Cuadrado"
        elif vertices > 4:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.75:
                shape = "Circulo"
            else:
                shape = "Figura desconocida"
        else:
            shape = "Figura desconocida"

        M = cv2.moments(contour)
        if M['m00'] != 0:  # que no sea 0
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv2.putText(frame, shape, (cx - 20, cy), font, 0.5, (0, 0, 0), 2)
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

    cv2.imshow("Figura Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
