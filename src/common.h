#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
using namespace std;

#include <opencv2\world.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\objdetect.hpp>
using namespace cv;

#include "json.h"
using namespace Json;

#include <io.h>
#include <direct.h>

#ifdef _DEBUG
const int nThreads = 1;
#else
const int nThreads = 3;
#endif // _DEBUG

const string dir_benchmark = "D:/face tracking/benchmark/";
const string dir_data = dir_benchmark + "data/";
const string dir_image = dir_benchmark + "image/";
const string path_groundtruth = dir_data + "groundtruth.json";
const string path_result = dir_data + "mt.json";

void mkdir(string dir);
Rect round(Rect2f rect);
float overlap(Rect2f a, Rect2f b);

class Detector
{
private:
	static mutex M_detect;
public:
	Detector();
	void detect(Mat gray, vector<Rect2f> &rects);

private:
	CascadeClassifier model;
};

class Sequence
{
public:
	static Sequence* next();
	static void finish(Sequence* seq);

public:
	Sequence(Value &_V);
	
	string image(int n);
	bool clear(int n);	
	Rect& rect(int n);		

public:	
	Value &V;
	string name;
	float rate;
	int start_frame, end_frame, width, height;
	string type;
	vector<Rect> rects;
};

class Statistics
{
public:
	Statistics(int isSeq);

	void frame(Rect gt, Rect ret, vector<Rect2f> *detections = NULL);

public:
	int nSeq, nFrame, nClear, nUnclear;
	int n50, n80, nDetect;
	int nDetectUnclear, nDetect50;
	double scores, scoresDetect, secs;
	clock_t start_clock;
};

ostream& operator<<(ostream& cout, const Statistics &st);
Statistics& operator+=(Statistics& st, const Statistics &opt);
