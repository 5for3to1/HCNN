#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<iomanip>

using namespace std;

void main()
{
	/*string in_base_path = "F:/MobileNetSSD/MobileNetSSD-DeepSense/l_";
	string out_base_path = "F:/MobileNetSSD/MobileNetSSD-DeepSense-android/l_";*/
	string in_base_path = "F:/MobileNetSSD/MobileNetSSD-face-DeepSense/l_";
	string out_base_path = "F:/MobileNetSSD/MobileNetSSD-face-DeepSense-android/l_";

	
	int N, C, K;

	for (int i = 1; i < 48; i++)
	{
		ostringstream os;
		os << i;
		string index_str = os.str();
		
		
		//read conv kernel size
		FILE * fp;
		fopen_s(&fp, (in_base_path + index_str).c_str(), "r");
		if (fp)
		{
			char line[256];
			fgets(line, sizeof(line), fp);
			fgets(line, sizeof(line), fp);
			fgets(line, sizeof(line), fp);
			fgets(line, sizeof(line), fp);
			sscanf_s(line, "WIDTH: %d\n", &K);
			fgets(line, sizeof(line), fp);
			sscanf_s(line, "HEIGHT: %d\n", &K);
			fgets(line, sizeof(line), fp);
			sscanf_s(line, "IN_CHANNELS: %d\n", &C);
			fgets(line, sizeof(line), fp);
			sscanf_s(line, "OUT_CHANNELS: %d\n", &N);
			cout << K << " " << C << " " << N << endl;
			fclose(fp);
		}

		//read bias
		float * bias = new float[N];
		ifstream read(in_base_path + index_str + "_bias");
		if (read)
		{
			for (int j = 0; j < N; j++)
			{
				read >> bias[j];
			}
			if (i==1)
			{
				for (int k = 0; k < N; k++)
				{
					cout << setprecision(7) << bias[k] << " ";
				}
				cout << endl << endl;
			}
		}
		read.close();

		//write constant bias
		fopen_s(&fp, (out_base_path + index_str + "_bias").c_str(), "wb");
		if (fp)
		{
			fwrite(bias, sizeof(float), N, fp);
			fclose(fp);
		}
		delete[] bias;

		//test read
		if (i==1)
		{
			fopen_s(&fp, (out_base_path + index_str + "_bias").c_str(), "rb");
			if (fp)
			{
				float * bias_result = new float[N];
				fread(bias_result, sizeof(float), N, fp);
				for (int k = 0; k < N; k++)
				{
					cout << setprecision(7) << bias_result[k] << " ";
				}
				cout << endl;
				delete[] bias_result;
				fclose(fp);
			}
		}


		//read weight
		float * weight = new float[N*C*K*K];
		read.open(in_base_path + index_str + "_weight");
		if (read)
		{
			for (int j = 0; j < N*C*K*K; j++)
			{
				read >> weight[j];
			}
		}
		read.close();

		//write constant weight
		fopen_s(&fp, (out_base_path + index_str + "_weight").c_str(), "wb");
		if (fp)
		{
			fwrite(weight, sizeof(float), N*C*K*K, fp);
			fclose(fp);
		}
		delete[] weight;

		cout << "finish conv" << i << endl;
	}

	system("pause");
}