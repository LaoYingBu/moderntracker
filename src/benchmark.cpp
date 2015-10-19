#include "common.h"
#include "mt.h"

namespace benchmark 
{
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
						}
					}
					ret = round(mt->track(gray, detections));
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

