#include "common.h"

rect_t cv2mt(Rect2f rect)
{
	rect_t _rect;
	_rect.x = rect.x;
	_rect.y = rect.y;
	_rect.width = rect.width;
	_rect.height = rect.height;
	return _rect;
}

Rect2f mt2cv(rect_t rect)
{
	Rect2f _rect;
	_rect.x = rect.x;
	_rect.y = rect.y;
	_rect.width = rect.width;
	_rect.height = rect.height;
	return _rect;
}

void mkdir(string dir)
{
#ifdef __linux
	if (access(dir.c_str(), F_OK) == -1)
		mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#else
	if (_access(dir.c_str(), 0) == -1)
		_mkdir(dir.c_str());
#endif // __linux
}

float overlap(Rect2f a, Rect2f b)
{
	if (a.width <= 0.0f || a.height <= 0.0f || a.area() <= 0.0f)
		return 0.0f;
	if (b.width <= 0.0f || b.height <= 0.0f || b.area() <= 0.0f)
		return 0.0f;
	float s = (a & b).area();
	return s / (a.area() + b.area() - s);
}

void detect(CascadeClassifier &detector, Mat gray, vector<Rect2f> &rects)
{
	vector<Rect> faces;
	detector.detectMultiScale(gray, faces, 1.1, 5);
	rects.clear();
	for (auto& r : faces) {
		float w = 0.78f * r.width;
		float h = 0.78f * r.height;
		float x = r.x + r.width * 0.5f - w * 0.5f;
		float y = r.y + r.height * 0.55f - h * 0.5f;
		rects.push_back(Rect2f(x, y, w, h));
	}
}

mutex Sequence::M_sequence;
Reader Sequence::reader;
Value Sequence::root;
vector<int> Sequence::perm;
vector<int>::iterator Sequence::perm_iter;

Sequence* Sequence::getSeq()
{
	M_sequence.lock();
	if (root.empty()) {
		ifstream fin(path_groundtruth);
		reader.parse(fin, root);
		fin.close();
		perm.resize(root.size());
		for (int i = 0; i < perm.size(); ++i)
			perm[i] = i;
		srand(0);
		random_shuffle(perm.begin(), perm.end());
		perm_iter = perm.begin();
	}
	Sequence *ret = NULL;
	if (perm_iter != perm.end()) {
		ret = new Sequence(root[*perm_iter]);
		++perm_iter;
	}
	else {		
		if (!path_result.empty()) {
			ofstream fout(path_result);
			fout << StyledWriter().write(root) << endl;
			fout.close();
		}
	}
	M_sequence.unlock();
	return ret;
}

void Sequence::setSeq(Sequence* seq)
{
	M_sequence.lock();
	Value &R = seq->V["rects"];
	auto iter = seq->rects.begin();
	for (auto &r : R) {
		r[0] = iter->x + 1;
		r[1] = iter->y + 1;
		r[2] = iter->width;
		r[3] = iter->height;
		++iter;
	}
	delete seq;
	M_sequence.unlock();
}

void Sequence::invoker()
{
	Sequence *seq = NULL;
	while ((seq = Sequence::getSeq()) != NULL) {
		if (seq->getType() == "whole") 
			seq->loadImage();
		Sequence::setSeq(seq);
	}	
}

Sequence::Sequence(Value &_V) : V(_V)
{				
	name = V["name"].asString();	
	start_frame = V["start_frame"].asInt();
	end_frame = V["end_frame"].asInt();
	width = V["width"].asInt();
	height = V["height"].asInt();
	type = V["type"].asString();
	image_scale = sqrt(float(resolution_width * resolution_height) / float(width * height));
	if (image_scale > 1.0f)
		image_scale = 1.0f;

	Value &R = V["rects"];	
	for (auto &r : R) {
		int x = r[0].asInt() - 1;
		int y = r[1].asInt() - 1;
		int w = r[2].asInt();
		int h = r[3].asInt();
		rects.push_back(Rect(x, y, w, h));
	}		
}

string Sequence::getName()
{
	return name;
}

int Sequence::getStart()
{
	return start_frame;
}

int Sequence::getEnd()
{		
	return end_frame;
}

int Sequence::getWidth()
{
	return int(width * image_scale + 0.5f);
}

int Sequence::getHeight()
{
	return int(height * image_scale + 0.5f);
}

string Sequence::getType()
{
	return type;
}

void Sequence::loadImage()
{
	if (!grays.empty())
		return;	
	for (int i = start_frame; i <= end_frame; ++i) {
		Mat t;
		stringstream ss;
		ss << dir_image << name << "/";
		ss.width(4);
		ss.fill('0');
		ss << i << "_" << getWidth() << "x" << getHeight() << ".jpg";		
#ifdef __linux
		if (access(ss.str().c_str(), F_OK) == -1) {			
#else
		if (_access(ss.str().c_str(), 0) == -1) {
#endif // __linux
			stringstream ss2;
			ss2 << dir_image << name << "/";
			ss2.width(4);
			ss2.fill('0');
			ss2 << i << ".jpg";
			Mat gray = imread(ss2.str(), 0);
			resize(gray, t, Size(getWidth(), getHeight()));			
			//imwrite(ss.str(), t);
		}
		else 
			t = imread(ss.str(), 0);	
		grays.push_back(t);
	}
}

Mat Sequence::getImage(int n)
{
	if (grays.empty())
		loadImage();
	return grays[n - start_frame];
}

bool Sequence::getClear(int n)
{
	return rects[n - start_frame].area() > 0;
}

Rect2f Sequence::getRect(int n)
{
	Rect rect = rects[n - start_frame];
	float x = rect.x * image_scale;
	float y = rect.y * image_scale;
	float width = rect.width * image_scale;
	float height = rect.height * image_scale;
	return Rect2f(x, y, width, height);
}

void Sequence::setRect(int n, Rect2f rect)
{
	int x = int(floor(rect.x / image_scale));
	int y = int(floor(rect.y / image_scale));
	int width = int(ceil((rect.x + rect.width) / image_scale)) - x;
	int height = int(ceil((rect.y + rect.height) / image_scale)) - y;
	rects[n - start_frame] = Rect(x, y, width, height);
}

Statistics::Statistics(bool isSeq)
{
	nSeq = int(isSeq);
	nFrame = nClear = nUnclear = 0;
	n50 = n80 = nSuccess = nFail = 0;
	nDetect = nCoarse = nMLK = nIteration = 0;
	scores = secs = 0.0;	
	start_clock = end_clock = 0.0;
}

void Statistics::tic()
{
	start_clock = double(getTickCount());
}

void Statistics::toc()
{
	end_clock = double(getTickCount());
	secs = (end_clock - start_clock) / getTickFrequency();
}

bool Statistics::empty()
{
	return nSeq == 0;
}

void Statistics::track(Rect2f gt, Rect2f result, bool success, int nDetect, int nCoarse, int nMLK, int nIteration)
{
	++nFrame;
	if (gt.area() > 0) {
		++nClear;
		float score = overlap(gt, result);
		if (score >= 0.5f)
			++n50;
		if (score >= 0.8f)
			++n80;
		scores += score;
		++(success ? nSuccess : nFail);
	}
	else
		++nUnclear;	
	this->nDetect += nDetect;
	this->nCoarse += nCoarse;
	this->nMLK += nMLK;
	this->nIteration += nIteration;
}

ostream& operator<<(ostream& cout, const Statistics &st)
{
	double p50 = double(st.n50) / double(st.nClear) * 100;
	double p80 = double(st.n80) / double(st.nClear) * 100;
	double pSuccess = double(st.nSuccess) / double(st.nClear) * 100;
	double pFail = double(st.nFail) / double(st.nClear) * 100;
	double pDetect = double(st.nDetect) / double(st.nFrame) * 100;
	double pCoarse = double(st.nCoarse) / double(st.nFrame) * 100;
	double avgMLK = double(st.nMLK) / double(st.nFrame);
	double avgIteration = double(st.nIteration) / double(st.nFrame);
	double avgIteration2 = double(st.nIteration) / double(st.nMLK);
	double avg = st.scores / max(st.nClear, 1);	
	if (st.nSeq > 1)
		cout << st.nSeq << " sequence(s) : " << endl;
	cout << "Clear / Total : " << st.nClear << "/" << st.nFrame << endl;
	cout << "Correct(>0.5) / Clear : " << st.n50 << "/" << st.nClear << "(" << p50 << "%)" << endl;
	cout << "Good(>0.8) / Clear : " << st.n80 << "/" << st.nClear << "(" << p80 << "%)" << endl;
	cout << "Success / Clear : " << st.nSuccess << "/" << st.nClear << "(" << pSuccess << "%)" << endl;
	cout << "Fail / Clear : " << st.nFail << "/" << st.nClear << "(" << pFail << "%)" << endl;	
	cout << "Average tracking overlap ratio : " << avg << endl;
	cout << "Re-detection / Total " << st.nDetect << "/" << st.nFrame << "(" << pDetect << "%)" << endl;
	cout << "Coarse search / Total : " << st.nCoarse << "/" << st.nFrame << "(" << pCoarse << "%)" << endl;
	cout << "Number of multi-scale Lucas-Kanade / Total : " << st.nMLK << "/" << st.nFrame << "(" << avgMLK << " every frame)" << endl;
	cout << "Number of iterations / Total : " << st.nIteration << "/" << st.nFrame << "(" << avgIteration << " every frame, " << avgIteration2 << " every MLK)" << endl;	
	cout << "Millisecond per frame : " << st.secs * 1000 / st.nFrame << endl;
	return cout;
}

Statistics& operator+=(Statistics& st, const Statistics &opt)
{
	st.nSeq += opt.nSeq;
	st.nFrame += opt.nFrame;
	st.nClear += opt.nClear;
	st.nUnclear += opt.nUnclear;
	st.n50 += opt.n50;
	st.n80 += opt.n80;
	st.nSuccess += opt.nSuccess;
	st.nFail += opt.nFail;
	st.nDetect += opt.nDetect;
	st.nCoarse += opt.nCoarse;
	st.nMLK += opt.nMLK;
	st.nIteration += opt.nIteration;	
	st.scores += opt.scores;	
	st.secs += opt.secs;
	return st;
}
