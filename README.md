# Face Tracking

Requirement:
1) opencv 3 or higher
2) eigen 3 or higher

Build:
please add "-O3 -msse2" for optimization, if possible

Run:
1) Make sure "haarcascade_frontalface_default.xml" and the binary are in the same folder
2) ./mt benchmark
	Test the algorithm on MFT120. You must specify the path of the benchmark in common.h
3) ./mt camera
	Real-time face tracking using a camera. You must have a available webcamera. 
4) ./mt example.avi
	face tracking on existing video. You must make sure the video is up-right.

Code:
main.cpp 		the entry
common.*		libraries and tool functions
jsoncpp.*		json library, used to parse MFT120 (slightly modified from JSONcpp)
benchmark.*		code to evaluate the tracker on MFT120
video.*			code to track face in a video or a camera
mt.*			the algorithm code
