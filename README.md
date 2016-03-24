Face Tracking

This is a face tracking demo on video by Mo Tao (mythly@qq.com)

Build:
1) install opencv 3.0 (the latest version until now, released in 2015.6.04)
2) add "opencv\build\include" in include path
3) add "opencv\build\**\**\lib" in library path
4) compile with the library "opencv_world300"

Run:
In the same directory with the binary 
1) make a copy of "haarcascade_frontalface_alt.xml"
2) build a subdirectory "result"
3) run "video_tracking.exe video_path [direction = 1]"
   a) video_path is the video you want to tracking
   b) direction is the orientation of the face, 1 for up 2 for left 3 for down 4 for right

Operation:
1) slide the trackbar to a frame such that the face you want to track is clear and frontal
2) modify the rectangle
   a) move it by wasd to keep the nose tip at center
   b) resize it by +- to capture the face properly
3) press enter to start tracking, it stops if 
   a) you press any key, except space (for pause)
   b) reach the end of the video
4) the result and log will be in the directory "/result"
5) press ESC to exit

Example:
(the win64 binary is prepared)
1) video_tracking rotate.MOV 1
2) switch to frame 41
3) adjust the box
4) start tracking
 
Use the code:
1) copy "mt.h" and "mt.cpp" to your project
2) include "mt.h"
3) call MT(gray scale image, bounding box of face, ostream of log) to construct an instance of face tracker
4) call track(gray scale image) to track a frame
5) MT.roll\yaw\pitch are euler angles
6) MT.error are tracking error
7) before 4) call suggest(possible bounding box of face) to give candidate to the tracker, when necessary


This is a real-time face tracking demo by Mo Tao (mythly@qq.com)

1. make sure your camera works normally
2. run camera_tracking.exe (only windows x64 supported now)
3. put your frontal face clearly in the camera view, and hold on
4. modify the rectangle
   a) move it by wasd to keep your nose tip at center
   b) resize it by +- to capture your face properly
5. press enter to start tracking, it stops if
   a) the tracker failed to track your face for longer than 5 second
   b) you press any key

FPS, face position, face size, roll, yaw and pitch angle are displayed in the command line

This is a face tracking demo on video by Mo Tao (mythly@qq.com)

Build:
1) install opencv 3.0 (the latest version until now, released in 2015.6.04)
2) add "opencv\build\include" in include path
3) add "opencv\build\**\**\lib" in library path
4) compile with the library "opencv_world300"

Run:
In the same directory with the binary 
1) make a copy of "haarcascade_frontalface_alt.xml"
2) build a subdirectory "result"
3) run "video_tracking.exe video_path [direction = 1]"
   a) video_path is the video you want to tracking
   b) direction is the orientation of the face, 1 for up 2 for left 3 for down 4 for right

Operation:
1) slide the trackbar to a frame such that the face you want to track is clear and frontal
2) modify the rectangle
   a) move it by wasd to keep the nose tip at center
   b) resize it by +- to capture the face properly
3) press enter to start tracking, it stops if 
   a) you press any key, except space (for pause)
   b) reach the end of the video
4) the result and log will be in the directory "/result"
5) press ESC to exit

Example:
(the win64 binary is prepared)
1) video_tracking rotate.MOV 1
2) switch to frame 41
3) adjust the box
4) start tracking
 
Use the code:
1) copy "mt.h" and "mt.cpp" to your project
2) include "mt.h"
3) call MT(gray scale image, bounding box of face, ostream of log) to construct an instance of face tracker
4) call track(gray scale image) to track a frame
5) MT.roll\yaw\pitch are euler angles
6) MT.error are tracking error
7) before 4) call suggest(possible bounding box of face) to give candidate to the tracker, when necessary
