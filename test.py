import os
import csv
import pickle
import base64
import argparse
import pandas as pd
import numpy as np
import progressbar


file = '/media/star/sdb/nesa/KDDCup2020/train/train.tsv'

df = pd.read_csv(file, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8', iterator=True)

flag = [False]*33
with progressbar.ProgressBar(max_value=3000000) as bar:
	count = 0
	while True:
		try:
			chunk = df.get_chunk(1000)
			count += len(chunk)
			for row in chunk.itertuples():
				labels = np.frombuffer(base64.b64decode(getattr(row, 'class_labels')), dtype=np.int64).reshape(getattr(row, 'num_boxes'))
				for label in labels:
					if not flag[label]:
						flag[label] = True
			bar.update(count)
		except StopIteration:
			break

for i in range(33):
	if not flag[i]:
		print('not exists {}'.format(i))