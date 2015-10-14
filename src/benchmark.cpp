#include "common.h"
#include "mt.h"

namespace benchmark 
{
	class Statistics
	{
	public:
		Statistics(int isSeq)
		{
			nSeq = isSeq;
			nFrame = nClear = nUnclear = 0;
			n50 = n80 = nDetect = 0;
			nDetectUnclear = nDetect50 = 0;
			scores = scoresDetect = secs = 0.0;
			start_clock = clock();
		}

		void frame(Rect gt, Rect ret, vector<Rect2f> *detections = NULL)
		{
			++nFrame;			
			if (gt.area() > 0) {
				++nClear;
				float score = overlap(gt, ret);
				if (score >= 0.5f)
					++n50;
				if (score >= 0.8f)
					++n80;
				scores += score;
			}
			if (detections != NULL) {
				++nDetect;
				if (gt.area() > 0) {										
					float score = 0.0f;
					for (auto r : *detections) 
					if (overlap(r, gt) > score)
						score = overlap(r, gt);
					if (score > 0.5f)
						++nDetect50;
					scoresDetect += score;
				}
				else
					++nDetectUnclear;				
			}						
			clock_t end_clock = clock();
			secs += double(end_clock - start_clock) / CLOCKS_PER_SEC;
			start_clock = end_clock;
		}

	public:
		int nSeq, nFrame, nClear, nUnclear;
		int n50, n80, nDetect;
		int nDetectUnclear, nDetect50;
		double scores, scoresDetect, secs;
		clock_t start_clock;
	};
	
	ostream& operator<<(ostream& cout, const Statistics &st)
	{		
		double p50 = double(st.n50) / double(st.nClear) * 100;
		double p80 = double(st.n80) / double(st.nClear) * 100;		
		if (st.nSeq > 1)
			cout << st.nSeq << " sequence(s) : " << endl;
		cout << "Clear / Total : " << st.nClear << "/" << st.nFrame << endl;
		cout << "Correct(>0.5) / Clear : " << st.n50 << "/" << st.nClear << "(" << p50 << "%)" << endl;		
		cout << "Good(>0.8) / Clear : " << st.n80 << "/" << st.nClear << "(" << p80 << "%)" << endl;		
		cout << "Average tracking overlap ratio : " << st.scores / st.nClear << endl;
		cout << "Detect / Total : " << st.nDetect << "/" << st.nFrame << endl;
		cout << "Detect(Unclear Detected Not detected) : ";
		cout << st.nDetectUnclear << " " << st.nDetect50 << " ";
		cout << st.nDetect - st.nDetectUnclear - st.nDetect50 << endl;
		if (st.nDetect - st.nDetectUnclear != 0)
			cout << "Average detection overlap ratio : " << st.scoresDetect / (st.nDetect - st.nDetectUnclear) << endl;		
		cout << "Frame per second : " << st.nFrame / st.secs << endl;
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
		st.nDetect += opt.nDetect;
		st.nDetectUnclear += opt.nDetectUnclear;			
		st.nDetect50 += opt.nDetect50;
		st.scores += opt.scores;
		st.scoresDetect += opt.scoresDetect;
		st.secs += opt.secs;
		return st;
	}

	const string dir_result = "./benchmark/";
	mutex M_global;
	Statistics global_short(false), global_long(false), global_whole(false);
	ofstream global_log;

	void invoker()
	{
		Sequence *seq = NULL;
		Detector detector;
		while ((seq = Sequence::next()) != NULL) {			
			stringstream ss;
			ss << "[" << seq->type << "]" << seq->name;
			ss << "(" << seq->start_frame << "-" << seq->end_frame << ")";
			ss << seq->width << "x" << seq->height;
			string alias = ss.str();
			
			M_global.lock();
			cout << alias << endl;
			M_global.unlock();
			
			ofstream log(dir_result + alias + ".txt");
			log << alias << endl;

			MT *mt = NULL;									
			Statistics st(true);
			vector<Rect2f> detections;

			for (int i = seq->start_frame; i <= seq->end_frame; ++i) {
				log << endl << "Frame " << i << " ";
				log << (seq->clear(i) ? "Clear" : "Unclear") << endl;
				Mat gray = imread(seq->image(i), 0);
				Rect gt = seq->rect(i), ret;
				bool detected = false;
				
				if (mt == NULL) 
					mt = new MT(gray, ret = gt, &log);
				else {
					if (seq->type != "short" && mt->miss()) {
						detected = true;
						detector.detect(gray, detections);
						log << detections.size() << " detections" << endl;
						for (auto r : detections) {
							log << "\t" << r;
							if (seq->clear(i))
								log << " " << overlap(r, gt);
							log << endl;
							mt->restart(r);
						}
					}
					ret = round(mt->track(gray));
				}
				log << "result : " << ret << endl;
				if (seq->clear(i)) 
					log << "groundtruth : " << gt << endl;
				log << "error : " << mt->error << endl;
				if (seq->clear(i)) {
					float score = overlap(gt, ret);
					log << "score : " << score << endl;
					if (score < 0.5f)
						log << "Wrong!!!" << endl;
				}

				st.frame(gt, ret, detected ? &detections : NULL);
				seq->rect(i) = ret;				
			}
			log << endl << st << endl;
			log.close();

			M_global.lock();
			global_log << alias << endl << st << endl;
			if (seq->type == "short")
				global_short += st;
			if (seq->type == "long")
				global_long += st;
			if (seq->type == "whole")
				global_whole += st;
			M_global.unlock();

			delete mt;
			seq->finish(seq);
		}
	}

	void main()
	{
		mkdir(dir_result);
		global_log.open(dir_result + "log.txt");

		vector<thread> ths;
		for (int i = 0; i < nThreads; ++i)
			ths.push_back(thread(&invoker));
		for (auto& th : ths)
			th.join();
				
		global_log << "Short benchmark : " << endl << global_short << endl;
		global_log << "Long benchmark : " << endl << global_long << endl;
		global_log << "Whole benchmark : " << endl << global_whole << endl;
		global_log.close();
	}
}

void run_benchmark()
{	
	cout << "Run benchmark" << endl;
	benchmark::main();
}

void run_unittest()
{
	cout << "Unit test" << endl;
}

void run_camera()
{
	cout << "Camera tracking" << endl;
}

void run_video(string path, int direction)
{
	cout << "Video tracking " << path << endl;
}
