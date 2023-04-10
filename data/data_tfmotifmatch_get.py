from pyjaspar import jaspardb
import numpy as np
from gtfparse import read_gtf
import gzip
import csv

'''
This file creates an array which maps each of the 2,753 regulatory proteins generated in
the data_tflevels_get file to their corresponding JASPAR motifs. Thus, we will have a
2,753 x 1,072 array. One may simply multiply the 2,753 expression values by this array and
the resulting array will correspond to the JASPAR motifs.

This file also creates a "bias array" which contains ones everywhere a JASPAR motif does not
correspond to any gene we have measured expressions for. This is not used later though.
'''

jdb_obj = jaspardb(release='JASPAR2022')
print(jdb_obj.release)
motifs = jdb_obj.fetch_motifs(all=True)
# Get all motifs for humans! There are 1237 (1,072 correspond to genes with measured expressions from GTEx)
humanMotifs = []
for motif in motifs:
    if motif.species == ['9606']:
        humanMotifs += [motif]
   
# Get list of tf express names (names of the transcription factors we have expression levels for, in order)
tfExpressNames = list(csv.reader(open("TFExpressNames.txt", 'r')))[0]
print(len(tfExpressNames))

df = read_gtf("gencode/gencode.v26.annotation.gtf")
# filter DataFrame to gene entries on chr1 through chrY
df = df[(df["feature"] == "gene") & ((df["seqname"] == 'chr1') | (df["seqname"] == 'chr2') | (df["seqname"] == 'chr3') | (df["seqname"] == 'chr4') |
                                                       (df["seqname"] == 'chr5') | (df["seqname"] == 'chr6') | (df["seqname"] == 'chr7') | (df["seqname"] == 'chr8') |
                                                       (df["seqname"] == 'chr9') | (df["seqname"] == 'chr10') | (df["seqname"] == 'chr11') | (df["seqname"] == 'chr12') |
                                                       (df["seqname"] == 'chr13') | (df["seqname"] == 'chr14') | (df["seqname"] == 'chr15') | (df["seqname"] == 'chr16') |
                                                       (df["seqname"] == 'chr17') | (df["seqname"] == 'chr18') | (df["seqname"] == 'chr19') | (df["seqname"] == 'chr20') |
                                                       (df["seqname"] == 'chr21') | (df["seqname"] == 'chr22') | (df["seqname"] == 'chrX') | (df["seqname"] == 'chrY'))]
   
tfExpressCommonNames = []

for idx, gene in df.iterrows():
    if gene.gene_id in tfExpressNames:
        tfExpressCommonNames += [gene.gene_name]

tfMotifMatchArray = np.zeros((2753, 1072))

cnt = 0
j = 0
for idx, gene in df.iterrows():
    if gene.gene_id in tfExpressNames:
        i = tfExpressNames.index(gene.gene_id)
        j = 0
        for motif in humanMotifs:
            if gene.gene_name == motif.name:
                tfMotifMatchArray[i, j] = 1
                cnt += 1
            if motif.name in tfExpressCommonNames:
                j += 1
print(cnt)
print(j)

tfBiasArray = np.zeros((1072))
for i in range(1072):
    if np.sum(tfMotifMatchArray[:, i]) == 0:
        tfBiasArray[i] = 1


f = gzip.GzipFile("tfMotifMatchArray.npy.gz", "w")
np.save(file=f, arr=tfMotifMatchArray)
f.close()

f = gzip.GzipFile("tfBiasArray.npy.gz", "w")
np.save(file=f, arr=tfBiasArray)
f.close()
