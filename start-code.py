import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from machathon_judge import Simulator, Judge

def smoothImage(image, kernelSize, sigma) :
    myImage = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    imageG = cv2.cvtColor(myImage,cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(imageG,kernelSize,sigma)
############
def getCannyImage(Simage) : 
    return cv2.Canny(image=Simage,threshold1= 100,threshold2= 220,apertureSize=3,L2gradient=True)
############
def getHouphLines(smoothImage) :
    canny = getCannyImage(smoothImage)
    return cv2.HoughLinesP(image=canny,rho=2,theta=np.pi/180,threshold=40,minLineLength=60,maxLineGap=50)
############
def splitImage(image) :
    h , w = image.shape
    left = image[:,:int(w/2)]
    right = image[:,int(w/2):]
    return (left,right)
def getBottomImage(image) :
    h , w = image.shape
    top = image[:int(3*h/4),:]
    bottom = image[int(3*h/4):,:]
    return (top,bottom)
def crop(image) :
    h , w = image.shape
    myImage = image[:,int(w/8):w-int(w/8)]
    return myImage
############
def getAVGdist(image):
    Simage = smoothImage(image,(9,9),3)
    top,bottom = getBottomImage(Simage)
    leftPart,rightPart = splitImage(bottom)
    leftLines = getHouphLines(leftPart)
    rightLines = getHouphLines(rightPart)
    height,width = leftPart.shape
    avgDistanceLeft=avgDistanceRight=0
    Tdistance = 0
    if leftLines is not None :
        pointOrigin=np.array((width,height))
        loops = 3 if len(leftLines)>3 else len(leftLines)
        for i in range(loops):
            line = leftLines[i][0]
            point=np.array((line[0],line[1]))
            dist = np.linalg.norm(pointOrigin-point)
            Tdistance += dist
        avgDistanceLeft = Tdistance/loops
    Tdistance = 0
    if rightLines is not None :
        pointOrigin=np.array((0,height))
        loops = 3 if len(rightLines)>3 else len(rightLines)
        for i in range(loops):
            line = rightLines[i][0]
            point=np.array((line[0],line[1]))
            dist = np.linalg.norm(pointOrigin-point)
            Tdistance += dist
        avgDistanceRight = Tdistance/loops
    return (avgDistanceLeft,avgDistanceRight)
############

# class FPSCounter:
#     def __init__(self):
#         self.frames = []

#     def step(self):
#         self.frames.append(time.monotonic())

#     def get_fps(self):
#         n_seconds = 5

#         count = 0
#         cur_time = time.monotonic()
#         for f in self.frames:
#             if cur_time - f < n_seconds:  # Count frames in the past n_seconds
#                 count += 1

#         return count / n_seconds


def run_car(simulator: Simulator) -> None:
    """
    Function to control the car using keyboard
    Parameters
    ----------
    simulator : Simulator
        The simulator object to control the car
        The only functions that should be used are:
        - get_image()
        - set_car_steering()
        - set_car_velocity()
        - get_state()
    """
    # fps_counter.step()

    # Get the image and show it
    image = simulator.get_image()
    # fps = fps_counter.get_fps()
    Dleft,Dright = getAVGdist(image)
    # print("Dleft="+str(Dleft) + "    "+"Dright="+str(Dright))
    # draw fps on image
    # cv2.putText(
    #     image,
    #     f"FPS: {fps:.2f}",
    #     (10, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0, 255, 0),
    #     2,
    #     cv2.LINE_AA,
    # )
    cv2.imshow("image", image)
    cv2.waitKey(1)
    v,s = simulator.get_state()
    # print(v)
    steering = 0
    throttle = 1
    # if Dleft < Dright+9 or Dleft > Dright+9 or Dleft < Dright-9 or Dleft > Dright+9 :
    #     throttle = 1
    if Dleft<Dright :
        if v>=15 :
            throttle = -1
        steering = 1
        print("Dleft < Dright")
    elif Dright < Dleft :
        if v>=15 :
            throttle = -1
        steering = -1
        print("Dright < Dleft")
    else :
        steering=0
        throttle = 1
    # if keyboard.is_pressed("a"):
    #     steering = 1
    # elif keyboard.is_pressed("d"):
    #     steering = -1

    
    # if keyboard.is_pressed("w"):
    #     throttle = 1
    # elif keyboard.is_pressed("s"):
    #     throttle = -1

    simulator.set_car_steering(steering * simulator.max_steer_angle /2)
    simulator.set_car_velocity(throttle * 2)

if __name__ == "__main__":
    # Initialize any variables needed
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # fps_counter = FPSCounter()

    # You should modify the value of the parameters to the judge constructor
    # according to your team's info
    judge = Judge(team_code="your_new_team_code", zip_file_path="your_solution.zip")

    # Pass the function that contains your main solution to the judge
    judge.set_run_hook(run_car)

    # Start the judge and simulation
    judge.run(send_score=False, verbose=True)