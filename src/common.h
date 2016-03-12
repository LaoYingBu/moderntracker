#ifndef common_h
#define common_h

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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#else
#include <io.h>
#include <direct.h>
#include <opencv2/world.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#endif /* __linux */

using namespace cv;

#ifndef Recf2f
#define Rect2f Rect_<float>
#endif  /* Rect2f */

#include "json.h"
using namespace Json;

void mkdir(string dir);
float overlap(Rect2f a, Rect2f b);

class Expr
{
public:
	Expr();
	void load(string path_configuration);
	string save();
	void edit(string param, string value);

private:
	void load();

public:
	//base configuration without modification
	string base_configuration;

	//make sure /data and /image are under this directory
	string dir_benchmark;
	string dir_data, dir_image, path_groundtruth;

	//path_log contatins the global statistics information	
	string path_log;
	//path_result contains the final rectangle output, use empty string to discard
	string path_result;
	//dir_detail contatins the log for each sequence, use empty string to discard
	string dir_detail;

	//number of threads, do not exceed the number of cores	
	int nThreads;
	//width and height can exchange, suggest 1280x720, 640x360 and 320x180
	int resolution_width, resolution_height;

	//only support Opencv pre-trained model		
	string detector_model;
	//The tracker call the detector if the average error over the last interval frames is lower than threshold	
	//If nothing detected, it will call the detector again after frequence frames
	float detector_threshold;
	int detector_interval, detector_frequence;	

	//fast template contatins about fast_n features	
	int fast_n;
	//the fast template moves at fast_step pixels
	int fast_step;
	//the search area of fast template is (1 + padding) * area_of_face
	float fast_padding;
	//when error is lower than threshold, carry out fast search
	float fast_threshold;

	//fine template contatins about fine_n features
	int fine_n;	
	//steps of multi-scale Lucas-Kanade, in decreasing order
	int fine_steps[4];
	//if the error is lower than threshold, update the fine model
	float fine_threshold;

	//each SURF feature is 4 cells, each cell is (2*cell_min)x(2*cell_min)	
	int cell_min;
	//each cell covers (area_of_face / cell_n) pixels
	int cell_n;

	//number of iteration of each scale
	int iteration_max;
	//termination condition of iteration
	float iteration_translate_eps, iteration_error_eps;

	//parameters of sigmoid
	float sigmoid_factor, sigmoid_bias;	

private:
	Reader reader;
	Value root;
};

extern Expr *expr;

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
	void track(Rect2f gt, Rect2f result, bool success, int number_MLK, int number_iteration);
	void fine_track(int choice, Rect2f gt, Rect2f start);
	void fast_track(int choice, Rect2f gt, Rect2f start);
	void detect_track(int choice, Rect2f gt, Rect2f start);

private:
	int nSeq, nFrame, nClear, nUnclear;
	int n50, n80, nSuccess, nFail;
	int nMLK, nIteration;
	int nFine, nFineClear, nFineUnclear, nFine50, nFineChoice;
	int nFast, nFastClear, nFastUnclear, nFast50, nFastChoice;
	int nDetect, nDetectClear, nDetectUnclear, nDetect50, nDetectChoice;	
	double scores, scoresFine, scoresFast, scoresDetect, secs;
	double start_clock, end_clock;
};

ostream& operator<<(ostream& cout, const Statistics &st);
Statistics& operator+=(Statistics& st, const Statistics &opt);

#endif /* common_h */
