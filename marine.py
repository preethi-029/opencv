1
"""import cv2
import numpy as np

cap = cv2.VideoCapture("marine.mp4")  # Use an underwater video
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for better underwater segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply background subtraction
    fgmask = fgbg.apply(hsv)

    # Noise reduction
    fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(fgmask, 50, 150)

    # Display results
    cv2.imshow('Foreground Mask', fgmask)
    cv2.imshow('Edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

2
"""import cv2
import numpy as np

cap = cv2.VideoCapture("marine.mp4")  # Use an underwater video
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Frame differencing
    diff = cv2.absdiff(prev_gray, gray)

    # Thresholding
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Noise reduction
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(thresh, 50, 150)

    # Display results
    cv2.imshow('Frame Difference', diff)
    cv2.imshow('Thresholded', thresh)
    cv2.imshow('Edges', edges)

    prev_gray = gray  # Update previous frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

3
import cv2
import numpy as np

cap = cv2.VideoCapture("marine.mp4")  # Use an underwater video
ret, frame = cap.read()

# Define the region of interest (ROI) for tracking
x, y, w, h = 200, 150, 50, 50  # Adjust based on object location
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Create histogram for tracking
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup Mean Shift tracking
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply Mean Shift
    ret, track_window = cv2.meanShift(dst, (x, y, w, h), term_crit)
    x, y, w, h = track_window

    # Draw tracking rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Tracked Object', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

4
"""import cv2
import numpy as np

cap = cv2.VideoCapture("fish.mp4")  # Use an underwater video
ret, frame = cap.read()

# Define the region of interest (ROI) for tracking
x, y, w, h = 200, 150, 50, 50  # Adjust based on object location
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Create histogram for tracking
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup CamShift tracking
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply CamShift
    ret, track_window = cv2.CamShift(dst, (x, y, w, h), term_crit)
    x, y, w, h = track_window

    # Draw tracking rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Tracked Object', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

5
"""import cv2
import numpy as np

cap = cv2.VideoCapture("marine.mp4")  # Use an underwater video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply blue filter
    blue_channel = frame[:, :, 0]  # Extract blue channel

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Display results
    cv2.imshow('Grayscale', gray)
    cv2.imshow('Blue Filter', blue_channel)
    cv2.imshow('Edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

