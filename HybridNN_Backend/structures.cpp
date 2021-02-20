#include <iostream>
#include <map>

using namespace std;

namespace structures {
class DatasetInfo {
	private:
		float* data;
		int columnSize;
		int rowSize;
	public:
		DatasetInfo() {
			this->data = NULL;
			this->columnSize = 0; 
			this->rowSize = 0;
		}
		DatasetInfo(float* data, int columnSize, int rowSize) {
			this->data = data;
			this->columnSize = columnSize;
			this->rowSize = rowSize;
		}
		int getColumnSize() {
			return this->columnSize;
		}
		int getRowSize() {
			return this->rowSize;
		}
		float getItem(int posX, int posY) {
			return this->data[posX * this->rowSize + posY];
		}
};

class DecisionTree {
	private:
		DatasetInfo datasetInfo;
		float entropy;
	public:
		DecisionTree(DatasetInfo datasetInfo) {
			this->datasetInfo = datasetInfo;
			this->calculateEntropy();
		}
		void calculateEntropy() {
			this->entropy = 0;
			map<float, int> frequencies;
			int target_feature_index = this->datasetInfo.getColumnSize() - 1;
			for (size_t row = 0; row < this->datasetInfo.getRowSize(); row++) {
				frequencies[datasetInfo.getItem(target_feature_index, row)]++;
		
			}
			map<float, int>::iterator it = frequencies.begin();
			while (it != frequencies.end()) {
				float p = (float)it->second / this->datasetInfo.getRowSize();
				if (p > 0) {
					this->entropy -= p * (log(p) / log(2));
				}
				it++;
			}
			cout << "entropy: " << this->entropy << endl;
		}
};
}