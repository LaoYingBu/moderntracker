#include "common.h"

namespace object
{
	const int threadNum = 1, visualizeID = -1;
	const string dir_object = "D:/face tracking/OOTB100/";
	const string dir_log = dir_object + "log/";
	const string path_log = dir_object + "log.txt";
	const string path_list = dir_object + "list.txt";
	const string path_face = dir_object + "face.txt";

	mutex M_global;
	ofstream global_log;
	vector< pair<string, set<string> > > sequences;	

	map<string, pair<int, double> > sum;	
	
	vector<Mat> loadImage(string sequence)
	{
		vector<Mat> ret;
		ret.push_back(Mat());
		for (int i = 1; i < 10000; ++i) {
			stringstream ss;
			ss << dir_object << sequence << "/img/";
			ss.width(4);
			ss.fill('0');
			ss << i << ".jpg";
			Mat img = imread(ss.str());
			if (img.empty())
				break;
			ret.push_back(img);
		}
		return ret;
	}

	vector<Rect> loadRectangle(string sequence) 
	{
		vector<Rect> ret;

		fstream fin(dir_object + sequence + "/groundtruth_rect.txt");
		string line;
		while (getline(fin, line)) {
			int x, y, w, h;
			sscanf(line.c_str(), "%d%*[^0-9]%d%*[^0-9]%d%*[^0-9]%d", &x, &y, &w, &h);
			ret.push_back(Rect(x - 1, y - 1, w, h));
		}
		return ret;
	}	

	void invoker(int threadID)
	{
		const string title = "tracking";
		if (threadID == visualizeID) {
			namedWindow(title);
			moveWindow(title, 1, 1);
		}

		CascadeClassifier detector(detector_model);
		
		for (int sequenceIdx = threadID; sequenceIdx < sequences.size(); sequenceIdx += threadNum) {
			string sequence = sequences[sequenceIdx].first;
			set<string> attributes = sequences[sequenceIdx].second;
			vector<Mat> imgs = loadImage(sequence);		
			vector<Rect> rects = loadRectangle(sequence), results;

			M_global.lock();
			cout << "thread" << threadID << ": " << sequence << " " << imgs.size() << endl;
			M_global.unlock();
			
			int startIdx = 1, endIdx = imgs.size() - 1;
			if (sequence == "David") {
				startIdx = 300;
				endIdx = 770;
			}
			if (sequence == "Football1")
				endIdx = 74;
			if (sequence == "Freeman3")
				endIdx = 460;
			if (sequence == "Freeman4")
				endIdx = 283;
			
			ofstream fout(dir_object + sequence + "/log.txt");
			MT *mt = NULL;
			int last_detect = attributes.find("FACE") != attributes.end() ? 0 : endIdx;

			for (int frameIdx = startIdx; frameIdx <= endIdx; ++frameIdx) {
				fout << endl << "Frame " << frameIdx << " ";
				Mat img = imgs[frameIdx], gray;				
				Rect rect = rects[frameIdx - startIdx], result;
				vector<Rect2f> detections;

				cvtColor(img, gray, CV_BGR2GRAY);
				if (mt == NULL) {
					result = rect;					
					mt = new MT(gray.data, gray.cols, gray.rows, cv2mt(result), &fout);
				}
				else {
					if (!mt->check() && frameIdx - last_detect >= detector_frequence) {
						detect(detector, gray, detections);
						last_detect = frameIdx;
					}
					if (detections.empty())
						result = mt2cv(mt->track(gray.data));
					else {
						vector<rect_t> mt_detections;
						for (auto d : detections)
							mt_detections.push_back(cv2mt(d));
						result = mt2cv(mt->retrack(gray.data, mt_detections));
					}
				}
				results.push_back(result);
				
				if (threadID == visualizeID) {
					rectangle(img, result, Scalar(0, 0, 255), 3);
					imshow(title, img);
					waitKey(10);
				}
			}
			delete mt;

			ofstream fresult(dir_object + sequence + "/result.txt");
			double AOS = 0;
			int N = rects.size();
			for (int i = 0; i < N; ++i) {				
				Rect r = results[i];
				AOS += overlap(rects[i], r);				
				fresult << r.x - 1 << "," << r.y - 1 << "," << r.width << "," << r.height << endl;
			}
			AOS /= N;
			
			M_global.lock();
			global_log << sequence << " " << N << " " << AOS << endl;
			cout << "thread" << threadID << " " << sequence << " " << N << " " << AOS << endl;
			for (auto attribute : attributes) {
				auto &t = sum.find(attribute) == sum.end() ? (sum[attribute] = make_pair(0, 0)) : sum[attribute];
				t.first += N;
				t.second += AOS * N;
			}			
			M_global.unlock();
		}
			
	}

	void main()
	{								
		fstream fin(path_list);
		string line;
		while (getline(fin, line)) {
			istringstream ss(line);
			string sequence, attribute;
			set<string> attributes;
			attributes.insert("OVERALL");
			ss >> sequence;
			while (ss >> attribute)
				attributes.insert(attribute);
			sequences.push_back(make_pair(sequence, attributes));
		}
		fin.close();	

		global_log.open(path_log, ios::out | ios::app);
		setNumThreads(0);

		
		double start_clock = double(getTickCount());		
		vector<thread> ths;
		for (int i = 0; i < threadNum; ++i)
			ths.push_back(thread(&invoker, i));
		for (auto& th : ths)
			th.join();
		double end_clock = double(getTickCount());
		double secs = (end_clock - start_clock) / getTickFrequency();

		global_log << "Time " << secs << " seconds" << endl;
		global_log << "FPS " << sum["OVERALL"].second / secs << endl;
		for (auto t : sum) {
			string attribute = t.first;
			int N = t.second.first;
			double AOS = t.second.second;
			global_log << attribute << " " << N << " " << AOS / N << endl;
		}		
	}
}

void run_object()
{
	cout << "Run object" << endl;
	object::main();
}
