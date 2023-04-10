import time
import pymysql
import numpy as np
import json
import urllib.request
from gtfparse import read_gtf
from Bio import SeqIO
import csv
import gzip
import pyBigWig
import matplotlib.pyplot as plt

'''
This file creates an array containing histone measurements for all 19786 genes,
each in the -2773 to +50 region, for three histone mark modifications, each measured
in 7 cell lines. This will form one input to the model.
'''

me1 = [pyBigWig.open("histones/H3k4me1/Gm12878H3k4me1.bigWig"),
       pyBigWig.open("histones/H3k4me1/H1hescH3k4me1.bigWig"),
       pyBigWig.open("histones/H3k4me1/HsmmH3k4me1.bigWig"),
       pyBigWig.open("histones/H3k4me1/HuvecH3k4me1.bigWig"),
       pyBigWig.open("histones/H3k4me1/K562H3k4me1.bigWig"),
       pyBigWig.open("histones/H3k4me1/NhekH3k4me1.bigWig"),
       pyBigWig.open("histones/H3k4me1/NhlfH3k4me1.bigWig")]

me3 = [pyBigWig.open("histones/H3k4me3/Gm12878H3k4me3.bigWig"),
       pyBigWig.open("histones/H3k4me3/H1hescH3k4me3.bigWig"),
       pyBigWig.open("histones/H3k4me3/HsmmH3k4me3.bigWig"),
       pyBigWig.open("histones/H3k4me3/HuvecH3k4me3.bigWig"),
       pyBigWig.open("histones/H3k4me3/K562H3k4me3.bigWig"),
       pyBigWig.open("histones/H3k4me3/NhekH3k4me3.bigWig"),
       pyBigWig.open("histones/H3k4me3/NhlfH3k4me3.bigWig")]

ac =  [pyBigWig.open("histones/H3k27ac/Gm12878H3k27ac.bigWig"),
       pyBigWig.open("histones/H3k27ac/H1hescH3k27ac.bigWig"),
       pyBigWig.open("histones/H3k27ac/HsmmH3k27ac.bigWig"),
       pyBigWig.open("histones/H3k27ac/HuvecH3k27ac.bigWig"),
       pyBigWig.open("histones/H3k27ac/K562H3k27ac.bigWig"),
       pyBigWig.open("histones/H3k27ac/NhekH3k27ac.bigWig"),
       pyBigWig.open("histones/H3k27ac/NhlfH3k27ac.bigWig")]

hm_bigwigs = [me1, me3, ac]

# Open the GENCODE annotations with essential columns such as "feature", "seqname", "start", "end"
# alongside the names of any optional keys which appeared in the attribute column
df = read_gtf("gencode/gencode.v26.annotation.gtf")
# filter DataFrame to gene entries on chr1 through chrY
df = df[(df["feature"] == "gene") & (df["gene_type"] == "protein_coding") & ((df["seqname"] == 'chr1') | (df["seqname"] == 'chr2') | (df["seqname"] == 'chr3') | (df["seqname"] == 'chr4') |
                                                       (df["seqname"] == 'chr5') | (df["seqname"] == 'chr6') | (df["seqname"] == 'chr7') | (df["seqname"] == 'chr8') |
                                                       (df["seqname"] == 'chr9') | (df["seqname"] == 'chr10') | (df["seqname"] == 'chr11') | (df["seqname"] == 'chr12') |
                                                       (df["seqname"] == 'chr13') | (df["seqname"] == 'chr14') | (df["seqname"] == 'chr15') | (df["seqname"] == 'chr16') |
                                                       (df["seqname"] == 'chr17') | (df["seqname"] == 'chr18') | (df["seqname"] == 'chr19') | (df["seqname"] == 'chr20') |
                                                       (df["seqname"] == 'chr21') | (df["seqname"] == 'chr22') | (df["seqname"] == 'chrX') | (df["seqname"] == 'chrY'))]



hmArr = np.zeros((19786, 2823, 3, 7))
with gzip.open('promoterData_gene.jsonl.gz', 'rt', encoding='UTF-8') as f:
    promoterDataIdx = 0
    for line in f:
        promoterData = json.loads(line)
        for i, tx in df[df['gene_id'] == promoterData['tx_ids'][0]].iterrows():
            chr = str(tx.seqname)

            if tx.strand == '+':
                tss = int(tx.start)
                tssPlus50 = int(tx.start) + 50
                tssMinus2773 = int(tx.start) - 2773
                for hm in range(3):
                    for cell in range(7):
                        if chr in hm_bigwigs[hm][cell].chroms():
                            hmVals =  np.array(hm_bigwigs[hm][cell].values(chr, tssMinus2773, tssPlus50))
                            hmVals[np.where(np.isnan(hmVals))] = 0
                            hmArr[promoterDataIdx, :, hm, cell] = hmVals
            elif tx.strand == '-':
                tss = int(tx.end)
                tssPlus50 = int(tx.end) - 50
                tssMinus2773 = int(tx.end) + 2773
                for hm in range(3):
                    for cell in range(7):
                        if chr in hm_bigwigs[hm][cell].chroms():
                            hmVals =  np.array(hm_bigwigs[hm][cell].values(chr, tssPlus50, tssMinus2773))
                            hmVals[np.where(np.isnan(hmVals))] = 0
                            hmVals = np.flip(hmVals)
                            hmArr[promoterDataIdx, :, hm, cell] = hmVals            

        promoterDataIdx += 1
        if promoterDataIdx % 100 == 0:
            print(promoterDataIdx)

f = gzip.GzipFile("hmArr.npy.gz", "w")
np.save(file=f, arr=hmArr)
f.close()