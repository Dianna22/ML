import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


TWITTER_RELATIVE_500 = "data/classification/Twitter/Relative_labeling/sigma=500/Twitter-Relative-Sigma-500.data"


def load_data():
	data = pd.read_csv(TWITTER_RELATIVE_500)
	return data
# return data[data.columns[:-1]], data[data.columns[-1]]



def class_imbalance():
	x,y = load_data()
	plt.hist(y)
	plt.show()


def playground():
	data = pd.read_csv(TWITTER_RELATIVE_500)

	print(data.columns)

	# plt.plot(data["NCD_0"])
	no_created_discussions = data[data.columns[:7]]
	# print(no_created_discussions)
	# plt.plot(no_created_discussions[1])
	# sns.distplot(data["NCD_0"]);
	plt.show()
	print(no_created_discussions.iloc[[0]].values)
	print(no_created_discussions.iloc[[0]].values)
	for i in range(no_created_discussions.shape[0])[:100]:
		plt.plot(no_created_discussions.iloc[[i]].values[0])
	plt.show()

print(class_imbalance())