from pyjaspar import jaspardb
import numpy as np
import gzip
import csv
from gtfparse import read_gtf

'''
This file creates two arrays:
1) An array containing the motifs of 13 core promoter elements.
2) An array containing the motifs for 1,072 transcription factors.

Each motif will be used as an untrainable convolutional kernel in the model.

This file also creates an array containing the frequency of observed binding events
for each motif, though this does not end up being used later.
'''

jdb_obj = jaspardb(release='JASPAR2020')
motifs = jdb_obj.fetch_motifs(all=True)

# First grab all motifs specifically for binding of the Polymerase 2 complex
POLIIMotifs = []
for motif in motifs:
    if motif.name == 'TATA-Box':
        print(motif)
    if motif.collection == 'POLII':
        POLIIMotifs += [motif]

jdb_obj = jaspardb(release='JASPAR2022')
motifs = jdb_obj.fetch_motifs(all=True)


# Grab motifs for human TFs
humanMotifs = []
for motif in motifs:
    if motif.species == ['9606']:
        humanMotifs += [motif]

# There are 1,072 binding motifs for humans matching the TFs we consider, and there are 13 motifs in the POLII set.
# The maximum motif length is 35 for the human set, so we will pad motifs with 0 so they are all 35.
# The maximum motif length is 19 for the POLII set, so we will pad motifs with 0 so they are all 19.
POLIIArray = np.zeros((19, 4, 13))
fullPSSMArray = np.zeros((35, 4, 1072))
fullCountArray = np.zeros((1072))

tfExpressNames = list(csv.reader(open("TFExpressNames.txt", 'r')))[0]
df = read_gtf("gencode/gencode.v26.annotation.gtf")
# filter DataFrame to gene entries on chr1 through chrY
df = df[(df["feature"] == "gene") & ((df["seqname"] == 'chr1') | (df["seqname"] == 'chr2') | (df["seqname"] == 'chr3') | (df["seqname"] == 'chr4') |
                                                       (df["seqname"] == 'chr5') | (df["seqname"] == 'chr6') | (df["seqname"] == 'chr7') | (df["seqname"] == 'chr8') |
                                                       (df["seqname"] == 'chr9') | (df["seqname"] == 'chr10') | (df["seqname"] == 'chr11') | (df["seqname"] == 'chr12') |
                                                       (df["seqname"] == 'chr13') | (df["seqname"] == 'chr14') | (df["seqname"] == 'chr15') | (df["seqname"] == 'chr16') |
                                                       (df["seqname"] == 'chr17') | (df["seqname"] == 'chr18') | (df["seqname"] == 'chr19') | (df["seqname"] == 'chr20') |
                                                       (df["seqname"] == 'chr21') | (df["seqname"] == 'chr22') | (df["seqname"] == 'chrX') | (df["seqname"] == 'chrY'))]
  

# Unordered list of the common names of my TFs I'm considering
tfExpressCommonNames = []

for idx, gene in df.iterrows():
    if gene.gene_id in tfExpressNames:
        tfExpressCommonNames += [gene.gene_name]


for i, motif in enumerate(POLIIMotifs):
    countArr = np.array(list(motif.counts.values())) + 2**-24

    pssmArr = np.log2((countArr / np.sum(countArr, axis=0)) / 0.25)
    if motif.name == 'TATA-Box':
        print(motif)
        print(pssmArr)
    motifLen = pssmArr.shape[1]
    dif = 19 - motifLen
    if dif % 2 == 0:
        pssmArr = np.pad(pssmArr, ((0, 0), (dif//2, dif//2)))
    else:
        pssmArr = np.pad(pssmArr, ((0, 0), (int(dif//2), int(dif//2) + 1)))
    POLIIArray[:, :, i] = pssmArr.T

i = 0
for motif in humanMotifs:
    if motif.name in tfExpressCommonNames:
        countArr = np.array(list(motif.counts.values())) + 2**-24
        pssmArr = np.log2((countArr / np.sum(countArr, axis=0)) / 0.25)

        motifLen = pssmArr.shape[1]
        dif = 35 - motifLen
        if dif % 2 == 0:
            pssmArr = np.pad(pssmArr, ((0, 0), (dif//2, dif//2)))
        else:
            pssmArr = np.pad(pssmArr, ((0, 0), (int(dif//2), int(dif//2) + 1)))
        
        fullPSSMArray[:, :, i] = pssmArr.T

        fullCountArray[i] = np.sum(list(motif.counts.values()), axis=0)[0]
        i += 1


f = gzip.GzipFile("POLIIKernels.npy.gz", "w")
np.save(file=f, arr=POLIIArray)
f.close()

f = gzip.GzipFile("PSSMKernels.npy.gz", "w")
np.save(file=f, arr=fullPSSMArray)
f.close()

f = gzip.GzipFile("PSSMCounts.npy.gz", "w")
np.save(file=f, arr=fullCountArray)
f.close()

print("done")