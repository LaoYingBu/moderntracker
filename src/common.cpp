#include "common.h"

void mkdir(string dir)
{
	if (_access(dir.c_str(), 0) == -1)
		_mkdir(dir.c_str());
}

Rect round(Rect2f rect)
{
	int x1 = int(floor(rect.x));
	int y1 = int(floor(rect.y));
	int x2 = int(ceil(rect.x + rect.width));
	int y2 = int(ceil(rect.y + rect.height));
	return Rect(x1, y1, x2 - x1, y2 - y1);
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

mutex Detector::M_detect;

Detector::Detector() : 
model("./haarcascade_frontalface_alt.xml")
{

}

void Detector::detect(Mat gray, vector<Rect2f> &rects)
{			
	M_detect.lock();
	vector<Rect> faces;
	model.detectMultiScale(gray, faces);	
	rects.clear();
	for (auto& r : faces) {
		float s = 0.78f;
		float w = r.width * s;
		float h = r.height * s;
		float x = r.x + r.width * 0.5f - w * 0.5f;
		float y = r.y + r.height * 0.55f - h * 0.5f;
		rects.push_back(Rect2f(x, y, w, h));
	}	
	M_detect.unlock();
}

mutex M_sequence;
Reader reader;
Value root;
Value::iterator root_iter;

Sequence* Sequence::next()
{
	M_sequence.lock();
	if (root.empty()) {
		ifstream fin(path_groundtruth);
		reader.parse(fin, root);
		fin.close();
		root_iter = root.begin();
	}
	Sequence *ret = NULL;
	if (root_iter != root.end()) {
		ret = new Sequence(*root_iter);
		++root_iter;
	}
	else {
		ofstream fout(path_result);
		fout << StyledWriter().write(root) << endl;
		fout.close();
	}
	M_sequence.unlock();
	return ret;
}

void Sequence::finish(Sequence* seq)
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
	rate = V["rate"].asFloat();
	start_frame = V["start_frame"].asInt();
	end_frame = V["end_frame"].asInt();
	width = V["width"].asInt();
	height = V["height"].asInt();
	type = V["type"].asString();
	Value &R = V["rects"];
	
	for (auto &r : R) {
		int x = r[0].asInt();
		int y = r[1].asInt();
		int w = r[2].asInt();
		int h = r[3].asInt();
		rects.push_back(Rect(x, y, w, h));
	}
}

string Sequence::image(int n)
{				
	stringstream ss;	
	ss << dir_image << name << "/";
	ss.width(4);
	ss.fill('0'); 
	ss << n << ".jpg";
	return ss.str();
}

bool Sequence::clear(int n)
{
	return rects[n - start_frame].area() > 0;
}

Rect& Sequence::rect(int n)
{
	return rects[n - start_frame];
}
