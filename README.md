# Face Tracking

This repo is the source code of the mobile face tracker and related demo code.

# Requirement

- opencv 3 or higher  
- eigen 3 or higher  

# Build

please add "-O3 -msse2" for optimization, if possible  

# Run

Make sure "haarcascade_frontalface_default.xml" and the binary are in the same folder  
- ./mt benchmark  
	Test the algorithm on MFT120. Make sure the benchmark is available. (Specify the path in common.h)  
- ./mt camera  
	Track a face in real-time using a camera. Make sure your webcamera is available.  
- ./mt example.avi  
	Track a face in existing video. Make sure the video is up-right.  

# Source code
under the folder /src

| file        | Content                                   |
|-------------|-------------------------------------------|
| main.cpp    | the entry                                 |
| common.*    | libraries and tool functions              |
| jsoncpp.*   | json library, used to parse MFT120        |
| benchmark.* | code to evaluate the tracker on MFT120    |
| video.*     | code to track face in a video or a camera |
| **mt.***    | **the algorithm code**                    |
