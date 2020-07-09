# Intellifotainment assist” – Smart HMI for passenger cars

To run the program, download all files and save them in the same directory.
After that, simply run 'Master.py' in the terminal.

At the moment, the program will only run in linux based systems.


 # The Idea
Infotainment systems have come a long way since the first set of dashboards installed in cars. Through our idea, we aim to create a Human Machine interaction model that takes infotainment systems to a new level. The driver tends to get distracted from the road while performing secondary tasks such as changing the music track, locking/unlocking the door while driving etc. Our system shall enable the driver to focus only on driving. Controlling the secondary tasks will be much easier.
Our product primarily comprises of 5 modules:

1)  Attention and drowsiness detection:
 A camera shall be present on the dashboard, in front of the driver, behind the steering wheel. Through digital image processing techniques , using hough circle algorithm and haarcascade of an eye, we shall keep track of the driver’s sight. If he or she is looking away from the road while driving for more than a specified amount of time, we shall alert the driver to focus. We shall map the head orientation and iris position to accurately identify the driver’s attention. 

2) Infotainment control features using blink combination:
Through a combination of blinks, the driver can turn on or of the headlights, tail lights as well as indicators. Blinking of the eyes shall be detected using ‘dlib’ features in python. This shall give extremely accurate results.


3)  Voice commands to control wipers, car lock, music system and windows
A simple, yet extremely useful idea that would make the life of the driver a whole lot easy. Enabling the driver to speak to his car infotainment system would allow him to control and navigate these functionalities with great ease. The car will be enabled with a virtual assistant.
 
 
4) Automatic rear view mirror adjustment scheme:
Using the camera placed in front of the driver, the system shall detect the position of the driver’s head. This shall also be done using image processing techniques and we shall identify the coordinates of the driver head in 3D space. 
There will be a mapping between the head position and mirror adjustment scheme. The mirrors will adjust their position using servo motors and shall do so automatically by identifying the head position. 

 
5)  A revolutionary reverse-assistance algorithm for smart parking and general reversing: Probably the highlight of our model, this feature shall make driving the car in and out of a parking spot, or rather, even reversing a car in general, far easier and safer than what it already is.
 
Like most other modern cars, our model shall also have a camera installed at the back and the corresponding image displayed at the infotainment screen for parking assistance. Upon activating the reverse gear, the screen shall trace the line of motion of the car corresponding to the current position of the steering wheel. Because of this feature, the driver gets an idea of whether or not he’ll hit an obstacle while reversing if the steering wheel is kept at that position.
 
Taking this feature to another level, the rear camera, after capturing the live video feed from the back of the car, shall perform image processing and machine learning algorithms to find a safe, obstacle-free path for reversing and indicate the driver to move the steering wheel accordingly. So instead of relying solely on the drivers judgement, our system shall actually find the path to be taken while reversing, such that other cars and other obstacles will be avoided, and accordingly recommend the driver to steer the wheel in that direction. This feature shall be extremely useful for new drivers/ learners.
 
During the initial phase, to prevent errors from creeping in, we will always have a manual override button. After a good amount of testing, further modifications and refinements can be made. Our systems adds new dimensions to both precautionary safety measures, as well as convenience. If implemented properly, we are confident that our project will reach new heights of HMI and driver assistance technology. It will give drivers several less reasons to worry about.
