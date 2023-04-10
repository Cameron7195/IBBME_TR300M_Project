import pandas as pd
import csv
import numpy as np
import json
from gtfparse import read_gtf
import gzip

'''
This file creates four things.
1) A dictionary mapping each GTEx sample id to a list of transcription factor levels.
2) A list of the names of all 2,753 transcription factors (the true transcription factors I use
   later are only 1,072 of these, but I include all 2,753 in case they might eventually become useful)
3) An array of 2,753 one-hot encoded chromosomes, indicating which chromosome each TF comes from.
4) An array of 2,753 tss values, indicating the transcriptional start site location of each TF-producing gene.

'''

# I think the bug is fixed... It was just an issue with not reading the correct headers (sample ids)
# But double check tomorrow...

def isGeneInTfList(gene_id, tfList):
    gene_id_beforeDecimal = ''
    for c in gene_id:
        if c != '.':
            gene_id_beforeDecimal += c
        else:
            break
    if gene_id_beforeDecimal in tfList:
        return True
    else:
        return False

def chrToOneHot(chr):
    chrIdx = {'chr1': 0,'chr2': 1,'chr3': 2,'chr4': 3,'chr5': 4,'chr6': 5,'chr7': 6,'chr8': 7,'chr9': 8,'chr10': 9,'chr11': 10,'chr12': 11,
                                'chr13': 12,'chr14': 13,'chr15': 14,'chr16': 15,'chr17': 16,'chr18': 17,'chr19': 18,'chr20': 19,'chr21': 20,'chr22': 21,
                                'chrX': 22,'chrY': 23}
    out = [0.0]*24
    out[chrIdx[chr]] = 1.0
    return out

xl = pd.read_excel(io="TF_List/TF_list.xlsx", sheet_name="Table S1. Related to Figure 1B")

tf_list = []
for i in range(3, 2766):
    tf_list += [str(xl.at[i, 'Gene Information'])]

print(len(tf_list))
tfNum = 0

df = read_gtf("gencode/gencode.v26.annotation.gtf")

# filter DataFrame to gene entries on chr1 through chrY
df = df[(df["feature"] == "gene") & ((df["seqname"] == 'chr1') | (df["seqname"] == 'chr2') | (df["seqname"] == 'chr3') | (df["seqname"] == 'chr4') |
                                                       (df["seqname"] == 'chr5') | (df["seqname"] == 'chr6') | (df["seqname"] == 'chr7') | (df["seqname"] == 'chr8') |
                                                       (df["seqname"] == 'chr9') | (df["seqname"] == 'chr10') | (df["seqname"] == 'chr11') | (df["seqname"] == 'chr12') |
                                                       (df["seqname"] == 'chr13') | (df["seqname"] == 'chr14') | (df["seqname"] == 'chr15') | (df["seqname"] == 'chr16') |
                                                       (df["seqname"] == 'chr17') | (df["seqname"] == 'chr18') | (df["seqname"] == 'chr19') | (df["seqname"] == 'chr20') |
                                                       (df["seqname"] == 'chr21') | (df["seqname"] == 'chr22') | (df["seqname"] == 'chrX') | (df["seqname"] == 'chrY'))]


# Read the gene_tpm file ***ONE*** row at a time so memory can survive.
tableHeaders = []
tfToExpress_dict = dict()
cnt = 0
with open("gtex/gene_tpm.gct") as infile:
    rows = csv.reader(infile, delimiter='\t')
    for rowElements in rows:
        if not isGeneInTfList(rowElements[0], tf_list) and cnt >= 3:
            continue
       
        if len(rowElements) != len(tableHeaders) and cnt >= 3:
            print("This shouldn't happpen")
            continue
        
        # This is just the headers row
        if cnt == 0:
            tableHeaders = rowElements
        
        # This is the table data now!
        if cnt >= 1:
            gene_id = rowElements[0]
            
            # Check if this gene is a transcription factor, if not: skip
            if isGeneInTfList(gene_id, tf_list):
                # Add expression profile to tfToExpress_dict
                tfToExpress_dict[gene_id] = np.array(rowElements[2:]).astype(float)
        
        cnt += 1
        print(cnt)
# There are 2753 Potential regulatory proteins from the excel sheet in the GTEx database,
# I will use this many in my model, however only 1,072 of these will be matched to TFs.
print(len(tfToExpress_dict))
sampleToTFExpress_dict = dict()
for i, sample_id in enumerate(tableHeaders):
    if i < 2: # First two columns are just gene_id and gene_name
        continue
    sampleToTFExpress_dict[sample_id] = []
    for gene_id, exProfiles in sorted(tfToExpress_dict.items()):
        sampleToTFExpress_dict[sample_id] += [exProfiles[i-2]]
    
tfExpressNames = [item[0] for item in sorted(tfToExpress_dict.items())]

tfChrArr = np.zeros((2753, 24, 1))
tfTssArr = np.zeros((2753, 1))
for idx, gene in df.iterrows():
    if gene.gene_id in tfExpressNames:
        i = tfExpressNames.index(gene.gene_id)
        if gene.strand == '+':
            tss = int(gene.start)
        elif gene.strand == '-':
            tss = int(gene.end)

        chrOneHot = chrToOneHot(gene.seqname)
        tfChrArr[i, :, :] = np.array(chrOneHot)[:, None]
        tfTssArr[i, :] = tss

with open('sampleToTFExpress.json', 'w') as fp:
    json.dump(sampleToTFExpress_dict, fp)

with open('TFExpressNames.txt', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(tfExpressNames)

f = gzip.GzipFile("tfChrArr.npy.gz", "w")
np.save(file=f, arr=tfChrArr)
f.close()

f = gzip.GzipFile("tfTssArr.npy.gz", "w")
np.save(file=f, arr=tfTssArr)
f.close()