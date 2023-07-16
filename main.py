import cv2
import numpy as np
import torch
import torchvision
import tensorflow
from deep_sort_realtime.deepsort_tracker import DeepSort
from kalman_filter import KalmanFilter
from detection import Detection
import iou_matching
import linear_assignment




# Initialize the object tracker
tracker = DeepSort(max_age=5) #If there is no detections on 5 frames in a row, the object is lost 
object_detector = Detection()
embedder = YourEmbedder()  # Replace with your own embedder, we need embedder 

# Initialize the Kalman filter
kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
#Set up the parameters for the Kalman filter (state transition matrix, measurement matrix)

# Start the video capture
video_capture = cv2.VideoCapture(0)  # 0 - camera source

while True:
    ret, frame = video_capture.read()
    
    # Object detection
    bbs = object_detector.detect(frame)
    object_chips = chipper(frame, bbs)  # Replace with our logic to crop frame based on bbox values
    embeds = embedder(object_chips)  # Replace with embedder
    
    # Kalman filter prediction
    predicted_bbs = []
    for track in tracker.tracks:
        if not track.is_confirmed():
            continue
        # Kalman filter prediction
        predicted_state = kalman_filter.predict()
        ltrb = track.to_ltrb()
        predicted_bb = [ltrb[0] + predicted_state[0],
                        ltrb[1] + predicted_state[1],
                        ltrb[2] + predicted_state[0],
                        ltrb[3] + predicted_state[1]]
        predicted_bbs.append(predicted_bb)

    # Hungarian algorithm for association
    if len(bbs) > 0 and len(predicted_bbs) > 0:
        iou_matrix = np.zeros((len(bbs), len(predicted_bbs)))
        for i, bb in enumerate(bbs):
            for j, predicted_bb in enumerate(predicted_bbs):
                iou =iou_matching(bb, predicted_bb)  # Replace with our IOU calculation function
                iou_matrix[i, j] = iou

        row_ind, col_ind = linear_assignment(-iou_matrix)
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] > threshold:  # Set a threshold to accept the association
                matched_bb = bbs[i]
                matched_track = tracker.tracks[j]
                # Update the Kalman filter with the matched detection
                kalman_filter.update(measurement=matched_bb[:2])
                kalman_filter.predict()
                # Update the track with the new detection
                matched_track.update(matched_bb, frame)
            else:
                # If the association is below the threshold, create a new track
                new_track = tracker.create_track(bbs[i], frame, embeds[i])

    # Display the results
    for track in tracker.tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        #Draw the bounding box and track ID on the frame

    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
    
    
    


