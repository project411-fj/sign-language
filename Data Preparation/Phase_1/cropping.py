import cv2
from ultralytics import YOLO
from tqdm import tqdm

def crop(path, progress_position=4):

    import os
    base_name = os.path.splitext(os.path.basename(path))[0]

    output_filename = f"cropvideos/{base_name}_output.avi"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_person = YOLO(os.path.join(BASE_DIR, "person.pt"))
    model_ties = YOLO(os.path.join(BASE_DIR, "ties.pt"))

    source = cv2.VideoCapture(path)

    total_frame = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(source.get(cv2.CAP_PROP_FPS))
    w = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    bar = tqdm(
        total=total_frame, desc="Phase 1 frames", unit="frame",
        position=progress_position, leave=False
    )

    while source.isOpened():
        success, frame = source.read()
        if success:

            results_person = model_person(frame,classes=[1],verbose=False)
            results_ties = model_ties(frame,verbose=False)

            has_person = len(results_person[0].boxes) > 0
            has_ties = len(results_ties[0].boxes) > 0

            if has_person and has_ties:

                out.write(frame)
            
            bar.update(1)
        else:
            break

    bar.close()
    source.release()
    out.release()
    cv2.destroyAllWindows()
    return output_filename
