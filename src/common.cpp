#include "common.h"

void mkdir(string dir)
{
#ifdef __linux__
	if (access(dir.c_str(), F_OK) == -1)
		mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#else
	if (_access(dir.c_str(), 0) == -1)
		_mkdir(dir.c_str());
#endif // __linux__
}

mutex M_detect;
CascadeClassifier detector(path_detector);

void detect(Mat gray, vector<Rect2f> &rects)
{
	M_detect.lock();
	vector<Rect> faces;
	detector.detectMultiScale(gray, faces);
	rects.clear();
	for (auto& r : faces) {
		float w = 0.78f * r.width;
		float h = 0.78f * r.height;
		float x = r.x + r.width * 0.5f - w * 0.5f;
		float y = r.y + r.height * 0.55f - h * 0.5f;
		rects.push_back(Rect2f(x, y, w, h));
	}
	M_detect.unlock();
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

mutex M_sequence;
Reader reader;
Value root;
vector<int> perm;
vector<int>::iterator perm_iter;

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
		ofstream fout(path_result);
		fout << StyledWriter().write(root) << endl;
		fout.close();
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
		r[0] = iter->x;
		r[1] = iter->y;
		r[2] = iter->width;
		r[3] = iter->height;
		++iter;
	}
	delete seq;
	M_sequence.unlock();
}

Sequence::Sequence(Value &_V) : V(_V)
{				
	name = V["name"].asString();	
	start_frame = V["start_frame"].asInt();
	end_frame = V["end_frame"].asInt();
	width = V["width"].asInt();
	height = V["height"].asInt();
	type = V["type"].asString();
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
		stringstream ss;
		ss << dir_image << name << "/";
		ss.width(4);
		ss.fill('0');
		ss << i << ".jpg";
		Mat gray = imread(ss.str(), 0), t;
		resize(gray, t, Size(getWidth(), getHeight()));
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
	n50 = n80 = nSuccess = nFail = nDetect = 0;
	nDetectUnclear = nDetect50 = 0;
	scores = scoresDetect = secs = 0.0;
	start_clock = end_clock = 0;
}

bool Statistics::empty()
{
	return nSeq == 0;
}

void Statistics::track(Rect2f gt, Rect2f result, bool success)
{
	if (nFrame == 0)
		start_clock = clock();
	end_clock = clock();
	secs = double(end_clock - start_clock) / CLOCKS_PER_SEC;

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
}

void Statistics::retrack(Rect2f gt, Rect2f result, bool success, const vector<Rect2f> &detections)
{
	track(gt, result, success);
	++nDetect;
	if (gt.area() > 0) {
		float score = 0.0f;
		for (auto r : detections)
		if (overlap(r, gt) > score)
			score = overlap(r, gt);
		if (score > 0.5f)
			++nDetect50;
		scoresDetect += score;
	}
	else
		++nDetectUnclear;
}

ostream& operator<<(ostream& cout, const Statistics &st)
{
	double p50 = double(st.n50) / double(st.nClear) * 100;
	double p80 = double(st.n80) / double(st.nClear) * 100;
	double pSuccess = double(st.nSuccess) / double(st.nClear) * 100;
	double pFail = double(st.nFail) / double(st.nClear) * 100;
	if (st.nSeq > 1)
		cout << st.nSeq << " sequence(s) : " << endl;
	cout << "Clear / Total : " << st.nClear << "/" << st.nFrame << endl;
	cout << "Correct(>0.5) / Clear : " << st.n50 << "/" << st.nClear << "(" << p50 << "%)" << endl;
	cout << "Good(>0.8) / Clear : " << st.n80 << "/" << st.nClear << "(" << p80 << "%)" << endl;
	cout << "Success / Clear : " << st.nSuccess << "/" << st.nClear << "(" << pSuccess << "%)" << endl;
	cout << "Fail / Clear : " << st.nFail << "/" << st.nClear << "(" << pFail << "%)" << endl;	
	cout << "Average tracking overlap ratio : " << st.scores / st.nClear << endl;
	cout << "Detect / Total : " << st.nDetect << "/" << st.nFrame << endl;
	cout << "Detect(Unclear Detected Not detected) : ";
	cout << st.nDetectUnclear << " " << st.nDetect50 << " ";
	cout << st.nDetect - st.nDetectUnclear - st.nDetect50 << endl;
	if (st.nDetect - st.nDetectUnclear != 0)
		cout << "Average detection overlap ratio : " << st.scoresDetect / (st.nDetect - st.nDetectUnclear) << endl;		
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
	st.nDetectUnclear += opt.nDetectUnclear;
	st.nDetect50 += opt.nDetect50;
	st.scores += opt.scores;	
	st.scoresDetect += opt.scoresDetect;
	st.secs += opt.secs;
	return st;
}
