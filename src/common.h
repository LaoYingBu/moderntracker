#ifndef common_h
#define common_h

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

#ifdef __linux
#include <unistd.h>
#include <sys/stat.h> 
#else
#include <io.h>
#include <direct.h>
#endif /* __linux */

//requires opencv3 or higher
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
using namespace cv;

#include "json.h"
using namespace Json;

const string dir_benchmark = "D:/face tracking/benchmark/";
const string dir_data = dir_benchmark + "data/";
const string dir_image = dir_benchmark + "image/";
const string path_groundtruth = dir_data + "groundtruth.json";
const string detector_model = "./haarcascade_frontalface_default.xml";
const int detector_frequence = 10;
const string dir_log = "./log/";
const string path_result = dir_log + "result.json";
const int resolution_width = 640, resolution_height = 360;

#include "mt.h"
rect_t cv2mt(Rect2f rect);
Rect2f mt2cv(rect_t rect);

void mkdir(string dir);
float overlap(Rect2f a, Rect2f b);
void detect(CascadeClassifier &detector, Mat gray, vector<Rect2f> &rects);

class Sequence
{
public:	
	static mutex M_sequence;
	static Reader reader;
	static Value root;
	static vector<int> perm;
	static vector<int>::iterator perm_iter;

public:	
	static Sequence* getSeq();
	static void setSeq(Sequence* seq);	

private:
	static void invoker();

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
	float image_scale;
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

	void tic();
	void toc();

	bool empty();
	void track(Rect2f gt, Rect2f result, bool success, int nDetect, int nCoarse, int nMLK, int nIteration);

private:
	int nSeq, nFrame, nClear, nUnclear;
	int n50, n80, nSuccess, nFail;
	int nDetect, nCoarse, nMLK, nIteration;
	double scores, secs;
	double start_clock, end_clock;
};

ostream& operator<<(ostream& cout, const Statistics &st);
Statistics& operator+=(Statistics& st, const Statistics &opt);

#endif /* common_h */
