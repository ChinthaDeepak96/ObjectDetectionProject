import cv2
import numpy as np

def region_of_interest(image):
    height, width = image.shape
    mask = np.zeros_like(image)

    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width / 2), int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked

def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def detect_lines(cropped_edges):
    return cv2.HoughLinesP(
        cropped_edges,
        rho=2,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

def draw_lines(frame, lines):
    if lines is None:
        return frame

    line_image = np.zeros_like(frame)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 8)

    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def lane_detection_main(video_source=0):
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = detect_edges(frame)
        cropped = region_of_interest(edges)
        lines = detect_lines(cropped)
        lane_frame = draw_lines(frame, lines)

        cv2.imshow("Lane Detection", lane_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lane_detection_main(0)
