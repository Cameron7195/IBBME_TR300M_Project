import time
import numpy as np
import json
import urllib.request
from gtfparse import read_gtf
from Bio import SeqIO
import csv
import gzip

'''
This file takes promoterData, sampleToCellType, and sampleToTFExpress, and reformats
them to be numpy representiations rather than dictionaries or lists. This is so that the
data examples may be generated by tensorflow 'on the fly' during training with optimal speed.
'''

'''
First reformat promeoterData and output (1) an array of creSequences, indexed by promoterIdx,
and (2) an array of acgtSequences, indexed by promoterIdx.
'''

with gzip.open('promoterData_gene.jsonl.gz', 'rt', encoding='UTF-8') as f:
    promoterDataIdx = 0
    creSeqArr = np.zeros((19786, 2823, 1))
    acgtSeqArr = np.zeros((19786, 2823, 4))
    chrArr = np.zeros((19786, 24, 1))
    tssArr = np.zeros((19786, 1))
    for line in f:
        promoterData = json.loads(line)
        creSeqArr[promoterDataIdx, :, :] = np.array(promoterData['creSeq'])[:, None]
        acgtSeqArr[promoterDataIdx, :, :] = np.array(promoterData['acgtOneHot'])
        chrArr[promoterDataIdx, :, :] = np.array(promoterData['chrOneHot'])[:, None]
        tssArr[promoterDataIdx, :] = np.array(promoterData['tss'])
        promoterDataIdx += 1
        if promoterDataIdx % 100 == 0:
            print(promoterDataIdx)

print(promoterDataIdx)

f = gzip.GzipFile("creSeqArr_gene.npy.gz", "w")
np.save(file=f, arr=creSeqArr)
f.close()
f = gzip.GzipFile("acgtSeqArr_gene.npy.gz", "w")
np.save(file=f, arr=acgtSeqArr)
f.close()
f = gzip.GzipFile("chrArr_gene.npy.gz", "w")
np.save(file=f, arr=chrArr)
f.close()
f = gzip.GzipFile("tssArr_gene.npy.gz", "w")
np.save(file=f, arr=tssArr)
f.close()


''' 
Next, reformat sampleToTFExpress and output (3) an array of tf gene expression levels, indexed by sampleIdx
Then, reformat sampleToCellType and output (4) an array of cell types, indexed by sampleIdx.
'''
sampleToTFExpress = json.loads(open('sampleToTFExpress.json', 'r').read())

sampleToCellType = json.loads(open('sampleToCellType.json', 'r').read())

tfExpressArr = np.zeros((17382, 2753, 1))
cellTypeArr = np.zeros((17382, 54, 1))

sampleIdx = 0
for sampleID, tfExpress in sampleToTFExpress.items():
    tfExpressArr[sampleIdx, :, :] = np.array(tfExpress)[:, None]
    sampleIdx += 1
print(sampleIdx)
print(tfExpressArr[17381])

f = gzip.GzipFile("tfExpressArr.npy.gz", "w")
np.save(file=f, arr=tfExpressArr)
f.close()


sampleIdx = 0
for sampleID, cellType in sampleToCellType.items():
    cellTypeArr[sampleIdx, :, :] = np.array(cellType)[:, None]
    sampleIdx += 1
print(sampleIdx)
print(cellTypeArr[17381])

f = gzip.GzipFile("cellTypeArr.npy.gz", "w")
np.save(file=f, arr=cellTypeArr)
f.close()
