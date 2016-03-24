# Face Tracking

Requirement: </br>
1) opencv 3 or higher </br>
2) eigen 3 or higher </br>

Build: </br>
please add "-O3 -msse2" for optimization, if possible </br>

Run: </br>
1) Make sure "haarcascade_frontalface_default.xml" and the binary are in the same folder </br>
2) ./mt benchmark </br>
	Test the algorithm on MFT120. You must specify the path of the benchmark in common.h </br>
3) ./mt camera </br>
	Real-time face tracking using a camera. You must have a available webcamera.  </br>
4) ./mt example.avi </br>
	face tracking on existing video. You must make sure the video is up-right. </br>

Code: </br>
main.cpp 		the entry </br>
common.*		libraries and tool functions </br>
jsoncpp.*		json library, used to parse MFT120 (slightly modified from JSONcpp) </br>
benchmark.*		code to evaluate the tracker on MFT120 </br>
video.*			code to track face in a video or a camera </br>
mt.*			the algorithm code </br>
