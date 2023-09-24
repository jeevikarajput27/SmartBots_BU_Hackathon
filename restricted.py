import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Define the vertical safety line
safety_line = [(300, 100), (300, 600)]  # Changed coordinates to make it vertical

detected_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    detected_faces = []

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        face_center_x = x + w // 2
        face_center_y = y + h // 2

        detected_faces.append((face_center_x, face_center_y))

    cv2.line(frame, safety_line[0], safety_line[1], (0, 255, 0), 2)

    for (center_x, center_y) in detected_faces:
        if center_x > safety_line[0][0]:  # Check if the face crosses the vertical line
            cv2.putText(frame, 'Danger!', (center_x - 30, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Safety Line and Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
