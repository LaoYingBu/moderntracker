#pragma once

#define _CRT_SECURE_NO_WARNINGS

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

#include <io.h>
#include <direct.h>

#include <opencv2\world.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\objdetect.hpp>
using namespace cv;

#include "json.h"
using namespace Json;

const string dir_benchmark = "D:/face tracking/benchmark/";
const string dir_data = dir_benchmark + "data/";
const string dir_image = dir_benchmark + "image/";
const string dir_common = "./common/";
const string path_detector = dir_common + "haarcascade_frontalface_default.xml";

#ifdef _DEBUG
const int nThreads = 1;
const string path_groundtruth = dir_common + "debug.json";
const string path_result = dir_common + "mt.json";
const float image_scale = 0.25f;
#else
const int nThreads = 3;
const string path_groundtruth = dir_data + "groundtruth.json";
const string path_result = dir_data + "mt.json";
const float image_scale = 0.5f;
#endif // _DEBUG

void mkdir(string dir);
void detect(Mat gray, vector<Rect2f> &rects);
float overlap(Rect2f a, Rect2f b);

class Sequence
{
public:
	static Sequence* getSeq();
	static void setSeq(Sequence* seq);

public:
	Sequence(Value &_V);
		
	string getName();
	int getStart();
	int getEnd();
	int getWidth();
	int getHeight();
	string getType();
	void loadImage();
	Mat getImage(int n);
	bool getClear(int n);
	Rect2f getRect(int n);
	void setRect(int n, Rect2f rect);

private:
	Value &V;
	string name;
	int start_frame, end_frame, width, height;
	string type;
	vector<Mat> grays;
	vector<Rect> rects;
};

class Statistics
{
	friend ostream& operator<<(ostream& cout, const Statistics &st);
	friend Statistics& operator+=(Statistics& st, const Statistics &opt);

public:
	Statistics(bool isSeq);

	bool empty();
	void track(Rect2f gt, Rect2f result, bool success);
	void retrack(Rect2f gt, Rect2f result, bool success, const vector<Rect2f> &detections);

private:
	int nSeq, nFrame, nClear, nUnclear;
	int n50, n80, nSuccess, nFail, nDetect;
	int nDetectUnclear, nDetect50;
	double scores, scoresDetect, secs;
	clock_t start_clock, end_clock;
};

ostream& operator<<(ostream& cout, const Statistics &st);
Statistics& operator+=(Statistics& st, const Statistics &opt);
