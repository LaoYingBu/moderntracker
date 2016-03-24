#include "common.h"

extern void run_benchmark();
extern void run_video(string path);

int main(int argc, char **argv)
{		
	if (argc != 2) {
		cerr << "test benchmark : mt benchmark" << endl;
		cerr << "test camera : mt camera" << endl;
		cerr << "test video : mt [path of video]" << endl;
		return 0;
	}
			
	string st(argv[1]);
	if (st == "benchmark")
		run_benchmark();
	else
		run_video(st);

	return 0;
}
