import torch
import cv2
import torchvision.transforms as transforms
import argparse
import time
from detection_utils import draw_bboxes
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', required=True, help='SSD_ResNet_video/20191202_032106.mp4'
)
args = vars(parser.parse_args())
# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# define the image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# initialize and set the model and utilities
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
ssd_model.to(device)
ssd_model.eval()
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
# capture the video
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                      (frame_width, frame_height))
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second
# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed_frame = transform(frame)
        tensor = torch.tensor(transformed_frame, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).to(device)
        # get the start time
        start_time = time.time()
        
        # get the detection results
        with torch.no_grad():
            detections = ssd_model(tensor)
        # the PyTorch SSD `utils` help get the detection for each input if...
        # ... there are more than one image in a batch
        # for us there is only one image per batch
        results_per_input = utils.decode_results(detections)
        # get all the results where detection threshold scores are >= 0.45
        # SSD `utils` help us here as well
        best_results_per_input = [utils.pick_best(results, 0.45) for results in results_per_input]
        # get the COCO object dictionary, again using `utils`
        classes_to_labels = utils.get_coco_object_dictionary()
        frame_result = draw_bboxes(frame, best_results_per_input, classes_to_labels)
        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # press `q` to exit
        wait_time = max(1, int(fps/4))
        # write the FPS on current frame
        cv2.putText(
            frame_result, f"{fps:.3f} FPS", (5, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
            2
        )
        cv2.imshow('image', frame_result)
        out.write(frame_result)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
