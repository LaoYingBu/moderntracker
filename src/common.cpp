#include "common.h"

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

Expr expr;

Expr::Expr()
{
	dir_benchmark = "D:/face tracking/benchmark/";
	dir_data = dir_benchmark + "data/";
	dir_image = dir_benchmark + "image/";
	path_groundtruth = dir_data + "groundtruth.json";

	path_log = "./benchmark/log.txt";
	path_result = dir_data + "mt.json";
	dir_detail = "./benchmark/";

	nThreads = 3;
	resolution_width = 640;
	resolution_height = 480;

	detector_model = "./common/haarcascade_frontalface_alt.xml";
	detector_threshold = 0.4f;
	detector_interval = 10;
	detector_frequence = 30;

	fast_n = 25;
	fast_step = 2;
	fast_padding = 1.6f;
	fast_threshold = 0.4f;

	fine_n = 600;
	fine_steps[0] = 27;
	fine_steps[1] = 9;
	fine_steps[2] = 3;
	fine_steps[3] = 1;
	fine_threshold = 0.4f;

	cell_min = 2;
	cell_n = 150;

	iteration_max = 9;
	iteration_translate_eps = 0.1f;
	iteration_error_eps = 0.001f;

	sigmoid_factor = 7.141f;
	sigmoid_bias = 0.482f;	

	configuration = "Default configuration ( see common.cpp::Expr::Expr() )";
}

void Expr::load(string _path_configuration)
{
	path_configuration = _path_configuration;
	path_configuration_old = _path_configuration;

	Reader reader;
	Value root;
	ifstream fin(path_configuration);
	reader.parse(fin, root);
	fin.close();

	dir_benchmark = root["dir_benchmark"].asString();
	dir_data = dir_benchmark + "data/";
	dir_image = dir_benchmark + "image/";
	path_groundtruth = dir_data + root["groundtruth"].asString();

	path_log = root["path_log"].asString();
	path_result = root["path_result"].asString();
	dir_detail = root["dir_detail"].asString();

	nThreads = root["nThreads"].asInt();
	resolution_width = root["resolution"]["width"].asInt();
	resolution_height = root["resolution"]["height"].asInt();

	detector_model = root["detector"]["model"].asString();		
	detector_threshold = root["detector"]["threshold"].asFloat();
	detector_interval = root["detector"]["interval"].asInt();
	detector_frequence = root["detector"]["frequence"].asInt();

	fast_n = root["fast"]["n"].asInt();
	fast_step = root["fast"]["step"].asInt();
	fast_padding = root["fast"]["padding"].asFloat();
	fast_threshold = root["fast"]["threshold"].asFloat();

	fine_n = root["fine"]["n"].asInt();
	fine_steps[0] = root["fine"]["steps"][0].asInt();
	fine_steps[1] = root["fine"]["steps"][1].asInt();
	fine_steps[2] = root["fine"]["steps"][2].asInt();
	fine_steps[3] = root["fine"]["steps"][3].asInt();
	fine_threshold = root["fine"]["threshold"].asFloat();

	cell_min = root["cell"]["min"].asInt();;
	cell_n = root["cell"]["n"].asInt();;

	iteration_max = root["iteration"]["max"].asInt();
	iteration_translate_eps = root["iteration"]["translate_eps"].asFloat();
	iteration_error_eps = root["iteration"]["error_eps"].asFloat();

	sigmoid_factor = root["sigmoid"]["factor"].asFloat();
	sigmoid_bias = root["sigmoid"]["bias"].asFloat();

	fin.open(path_configuration);
	configuration = string((istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
	fin.close();
}

void Expr::edit(string param, string value)
{
	if(param == "fine_steps")
	{
		vector<int> v(4);
		int pos1 = 0, pos2;
		for(int i = 0; i < 3; i++)
		{
			pos2 = value.find(",");
			stringstream ss;
			ss << value.substr(pos1, pos2 - pos1);
			ss >> v[i];
			pos1 = pos2 + 1;
		}
		stringstream ss;
		ss << value.substr(pos1, value.length());
		ss >> v[3];

		edit(param, v);
	}
	else
	{
		stringstream ss;
		ss << value;
		double value_d;
		ss >> value_d;

		edit(param, value_d);
	}
	path_log = path_log.substr(0, path_log.length() - 4) + "_(" + param + ":" + value + ").txt";
	path_configuration = path_configuration.substr(0, path_log.length() - 5) + "_(" + param + ":" + value + ").json";

}

void Expr::edit(string param, vector<int> value)
{
	if(param == "fine_steps")
	{
		for(int i = 0; i < value.size(); i++)
		{
			fine_steps[i] = value[i];
		}
	}
	else
	{
		fprintf(stderr, "%s\n", "No such param or type of the value dose not match the param.");
		return;
	}
}

void Expr::edit(string param, double value)
{
	if(param == "resolution_width")
			resolution_width = int(value);

		else if(param == "resolution_height")
			resolution_height = int(value);

		else if(param == "detector_threshold")
			detector_threshold = value;

		else if(param == "detector_interval")
			detector_interval = int(value);

		else if(param == "detector_frequence")
			detector_frequence = int(value);

		else if(param == "fast_n")
			fast_n = int(value);

		else if(param == "fast_step")
			fast_step = int(value);

		else if(param == "fast_padding")
			fast_padding = value;

		else if(param == "fast_threshold")
			fast_threshold = value;

		else if(param == "fine_n")
			fine_n = int(value);

		else if(param == "fine_threshold")
			fine_threshold = value;

		else if(param == "cell_min")
			cell_min = int(value);

		else if(param == "cell_n")
			cell_n = int(cell_n);

		else if(param == "iteration_max")
			iteration_max = int(value);

		else if(param == "iteration_translate_eps")
			iteration_translate_eps = value;

		else if(param == "iteration_error_eps")
			iteration_error_eps = value;

		else if(param == "sigmoid_factor")
			sigmoid_factor = value;

		else if(param == "sigmoid_bias")
			sigmoid_bias = value;
		
		else
			fprintf(stderr, "%s\n", "No such param or type of the value dose not match the param.");

}

void Expr::save()
{
	Reader reader;
	Value root;
	ifstream fin(path_configuration_old);
	reader.parse(fin, root);
	fin.close();

	root["path_log"] = path_log;

	root["resolution"]["width"] = resolution_width;
	root["resolution"]["height"] = resolution_height;

	root["detector"]["threshold"] = detector_threshold;
	root["detector"]["interval"] = detector_interval;
	root["detector"]["frequence"] = detector_frequence;

	root["fast"]["n"] = fast_n;
	root["fast"]["step"] = fast_step;
	root["fast"]["padding"] = fast_padding;
	root["fast"]["threshold"] = fast_threshold;

	root["fine"]["n"] = fine_n;
	root["fine"]["steps"][0] = fine_steps[0];
	root["fine"]["steps"][1] = fine_steps[1];
	root["fine"]["steps"][2] = fine_steps[2];
	root["fine"]["steps"][3] = fine_steps[3];
	root["fine"]["threshold"] = fine_threshold;

	root["cell"]["min"] = cell_min;
	root["cell"]["n"] = cell_n;

	root["iteration"]["max"] = iteration_max;
	root["iteration"]["translate_eps"] = iteration_translate_eps;
	root["iteration"]["error_eps"] = iteration_error_eps;

	root["sigmoid"]["factor"] = sigmoid_factor;
	root["sigmoid"]["bias"] = sigmoid_bias;

	ofstream fout(path_configuration);
	fout << StyledWriter().write(root) << endl;
	fout.close();

	fin.open(path_configuration);
	configuration = string((istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
	fin.close();
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
		ifstream fin(expr.path_groundtruth);
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
		if (!expr.path_result.empty()) {
			ofstream fout(expr.path_result);
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

Sequence::Sequence(Value &_V) : V(_V)
{				
	name = V["name"].asString();	
	start_frame = V["start_frame"].asInt();
	end_frame = V["end_frame"].asInt();
	width = V["width"].asInt();
	height = V["height"].asInt();
	type = V["type"].asString();
	image_scale = sqrt(float(expr.resolution_width * expr.resolution_height) / float(width * height));
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
		stringstream ss;
		ss << expr.dir_image << name << "/";
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
	n50 = n80 = nSuccess = nFail = 0;
	nMLK = nIteration = 0;
	nFine = nFineClear = nFineUnclear = nFine50 = nFineChoice = 0;
	nFast = nFastClear = nFastUnclear = nFast50 = nFastChoice = 0;
	nDetect = nDetectClear = nDetectUnclear = nDetect50 = nDetectChoice = 0;
	scores = scoresFine = scoresFast = scoresDetect = secs = 0.0;	
	start_clock = end_clock = 0;
}

bool Statistics::empty()
{
	return nSeq == 0;
}

void Statistics::track(Rect2f gt, Rect2f result, bool success, int number_MLK, int number_iteration)
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
	else
		++nUnclear;
	nMLK += number_MLK;
	nIteration += number_iteration;
}

void Statistics::fine_track(int choice, Rect2f gt, Rect2f start)
{
	++nFine;
	if (gt.area() > 0) {
		++nFineClear;
		float score = overlap(gt, start);
		if (score >= 0.5f)
			++nFine50;
		scoresFine += score;
		if (choice == 1)
			++nFineChoice;
	}
	else
		++nFineUnclear;	
}

void Statistics::fast_track(int choice, Rect2f gt, Rect2f start)
{
	++nFast;
	if (gt.area() > 0) {
		++nFastClear;
		float score = overlap(gt, start);
		if (score >= 0.5f)
			++nFast50;
		scoresFast += score;
		if (choice == 2)
			++nFastChoice;
	}
	else
		++nFastUnclear;
}

void Statistics::detect_track(int choice, Rect2f gt, Rect2f start)
{
	++nDetect;
	if (gt.area() > 0) {
		++nDetectClear;
		float score = overlap(gt, start);
		if (score >= 0.5f)
			++nDetect50;
		scoresDetect += score;
		if (choice == 3)
			++nDetectChoice;
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
	double avgMLK = double(st.nMLK) / double(st.nFrame);
	double avgIteration = double(st.nIteration) / double(st.nFrame);
	double avgIteration2 = double(st.nIteration) / double(st.nMLK);
	double avg = st.scores / max(st.nClear, 1);
	double avgFine = st.scoresFine / max(st.nFineClear, 1);
	double avgFast = st.scoresFast / max(st.nFastClear, 1);
	double avgDetect = st.scoresDetect / max(st.nDetectClear, 1);
	if (st.nSeq > 1)
		cout << st.nSeq << " sequence(s) : " << endl;
	cout << "Clear / Total : " << st.nClear << "/" << st.nFrame << endl;
	cout << "Correct(>0.5) / Clear : " << st.n50 << "/" << st.nClear << "(" << p50 << "%)" << endl;
	cout << "Good(>0.8) / Clear : " << st.n80 << "/" << st.nClear << "(" << p80 << "%)" << endl;
	cout << "Success / Clear : " << st.nSuccess << "/" << st.nClear << "(" << pSuccess << "%)" << endl;
	cout << "Fail / Clear : " << st.nFail << "/" << st.nClear << "(" << pFail << "%)" << endl;	
	cout << "Average tracking overlap ratio : " << avg << endl;
	cout << "Number of multi-scale Lucas-Kanade / Total : " << st.nMLK << "/" << st.nFrame << "(" << avgMLK << " every frame)" << endl;
	cout << "Number of iterations / Total : " << st.nIteration << "/" << st.nFrame << "(" << avgIteration << " every frame, " << avgIteration2 << " every MLK)" << endl;
	cout << "Initial times (Fine Fast Detect) " << st.nFine << " " << st.nFast << " " << st.nDetect << endl;
	cout << "Initial overlap ratio (Fine Fast Detect) : " << avgFine << " " << avgFast << " " << avgDetect << endl;
	cout << "Final choice (Fine Fast Detect) : " << st.nFineChoice << " " << st.nFastChoice << " " << st.nDetectChoice << endl;	
	cout << "Fine initial (Good Bad Unclear) : " << st.nFine50 << " " << st.nFineClear - st.nFine50 << " " << st.nFineUnclear << endl;
	cout << "Fast initial (Good Bad Unclear) : " << st.nFast50 << " " << st.nFastClear - st.nFast50 << " " << st.nFastUnclear << endl;
	cout << "Detect initial (Good Bad Unclear) : " << st.nDetect50 << " " << st.nDetectClear - st.nDetect50 << " " << st.nDetectUnclear << endl;			
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
	st.nMLK += opt.nMLK;
	st.nIteration += opt.nIteration;
	st.nFine += opt.nFine;
	st.nFineClear += opt.nFineClear;
	st.nFineUnclear += opt.nFineUnclear;
	st.nFine50 += opt.nFine50;
	st.nFineChoice += opt.nFineChoice;
	st.nFast += opt.nFast;
	st.nFastClear += opt.nFastClear;
	st.nFastUnclear += opt.nFastUnclear;
	st.nFast50 += opt.nFast50;
	st.nFastChoice += opt.nFastChoice;
	st.nDetect += opt.nDetect;
	st.nDetectClear += opt.nDetectClear;
	st.nDetectUnclear += opt.nDetectUnclear;
	st.nDetect50 += opt.nDetect50;
	st.nDetectChoice += opt.nDetectChoice;
	st.scores += opt.scores;
	st.scoresFine += opt.scoresFine;
	st.scoresFast += opt.scoresFast;
	st.scoresDetect += opt.scoresDetect;
	st.secs += opt.secs;
	return st;
}
