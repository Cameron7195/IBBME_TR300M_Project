import time
import pymysql
import numpy as np
import json
import urllib.request
from gtfparse import read_gtf
from Bio import SeqIO
import csv
import gzip

'''
This file creates a list of dictionaries and stores them in promoterDataDict.jsonl.gz

Each dictionary has as its key: chr-tssMinus2773-tssPlus50 
and contains the value: [txIds, seq, cres, chr, tss]

txIds is a list of ENCODE gene names (usually length 1) for the transcriptional start site.
seq is a one-hot encoded representation of the -2773 to +50 base pair DNA sequence.
cres contains measurements of Candidate Cis-Regulatory Regions from -2773 to +50 base pairs.
chr is a one-hot encoded representation of the chromosome where this gene is located.
tss is the transcriptional start site in the hg38 reference genome of this gene.
'''

#Set the hostIP, username, db, pwd.  Defaults are hostIP=genome-mysql.soe.ucsc.edu',db='hg38', username = 'genomep',pwd ='password'
#mysqlHostIP="localhost"
mysqlHostIP="apel.ibbme.utoronto.ca"
db="hg38"
username="genomep"
pwd ="password"

# Define an object to handle web requests to the UCSC database.
class ucscDbAPI:
  cursor=None
  def __init__(self,hostIP='genome-mysql.soe.ucsc.edu',db='hg38', username = 'genomep',pwd ='password'):
    self.hostIP=hostIP
    self.db=db
    self.username=username
    self.pwd=pwd

    self.db = pymysql.connect(host=self.hostIP, user=self.username, password=self.pwd, db=self.db, port=3306)
    self.cursor = self.db.cursor()

# Defiine a function to execute SQL queries.
def runQuery (ucscDb,query):
  ucscDb.cursor.execute(query)
  row_headers=[x[0] for x in ucscDb.cursor.description]
  finalList=[]
  for x in ucscDb.cursor:
    xD = dict(zip(row_headers,x))
    finalList.append(xD)
  return finalList
    
# Return a table of all Cis-Regulatory Elements
def getEncodeCreTable():
    query = """
    select * from hg38.encodeccrecombined
    """
    result = runQuery (ucscDb=ucscDb,query=query)
    return result
    
# Define a function to return a string representation of any subsequence of the hg38 reference genome.
def getSeq(genomeRecords, chrom, chromStart, chromEnd):
    chrIdx = {'chr1': 0,'chr2': 1,'chr3': 2,'chr4': 3,'chr5': 4,'chr6': 5,'chr7': 6,'chr8': 7,'chr9': 8,'chr10': 9,'chr11': 10,'chr12': 11,
                                    'chr13': 12,'chr14': 13,'chr15': 14,'chr16': 15,'chr17': 16,'chr18': 17,'chr19': 18,'chr20': 19,'chr21': 20,'chr22': 21,
                                    'chrX': 22,'chrY': 23}
    if chromEnd < chromStart:
        return "ERROR!"
    return genomeRecords[chrIdx[chrom]].seq[chromStart:chromEnd]

# Define a function to return the sequence of cre measurements for any subsection of the hg38 reference genome.          
def getCreSeq(creTable, chrom, tss, promEnd):
    cCRESequence = [0.0]*abs(promEnd-tss)
    for row in creTable:
        if row['chrom'] != chrom:
            continue
        chromSt = row['chromStart']
        chromEnd = row['chromEnd']
        
        if chromEnd < min(tss, promEnd) or chromSt > max(tss, promEnd):
            continue
            
        # If we get here, then cre impacts promoter!
        ccreSeqIdx = 0
        for chromIdx in range(min(tss, promEnd), max(tss, promEnd)):
            if chromIdx >= chromSt and chromIdx < chromEnd:
                cCRESequence[ccreSeqIdx] = row['score']
                #print(row['score'])
            ccreSeqIdx += 1
    return cCRESequence
    
# Define a function to return the reverse complement of a string-represented DNA sequence
def reverseComplement(seq):
    complementSeq = ''
    
    # First complement
    for letter in seq:
        if letter == 'A' or letter == 'a':
            complementSeq += 'T'
        elif letter == 'T' or letter == 't':
            complementSeq += 'A'
        elif letter == 'G' or letter == 'g':
            complementSeq += 'C'
        elif letter == 'C' or letter == 'c':
            complementSeq += 'G'
        elif letter == 'N' or letter == 'n':
            complementSeq += 'N'
        else:
            print("Weird letter detected....")
    
    retSeq = ''
    for letter in complementSeq:
        retSeq = letter + retSeq
    
    if len(seq) != len(retSeq):
        print("PROBLEM! Length changed")

    return retSeq
    
# Define a function to return a list in the opposite order.
def reverseList(l):
    out = []
    for item in l:
        out = [item] + out
    return out

# Define a function to convert a string DNA sequence to its one-hot encoding.
def seqToOneHot(seq):
    n = len(seq)
    out = [[0.0]*4 for i in range(2823)]
    for i, letter in enumerate(seq):
        if letter == 'A':
            out[i] = [1, 0, 0, 0]
        elif letter == 'C':
            out[i] = [0, 1, 0, 0]
        elif letter == 'G':
            out[i] = [0, 0, 1, 0]
        elif letter == 'T':
            out[i] = [0, 0, 0, 1]
        elif letter == 'N':
            out[i] = [0.25, 0.25, 0.25, 0.25]
        else:
            print("This shouldn't happen...")
    return out

# Define a function to convert a chromosome, represented as a string, to its one-hot encoding.
def chrToOneHot(chr):
    chrIdx = {'chr1': 0,'chr2': 1,'chr3': 2,'chr4': 3,'chr5': 4,'chr6': 5,'chr7': 6,'chr8': 7,'chr9': 8,'chr10': 9,'chr11': 10,'chr12': 11,
                                'chr13': 12,'chr14': 13,'chr15': 14,'chr16': 15,'chr17': 16,'chr18': 17,'chr19': 18,'chr20': 19,'chr21': 20,'chr22': 21,
                                'chrX': 22,'chrY': 23}
    out = [0.0]*24
    out[chrIdx[chr]] = 1.0
    return out


ucscDb =  ucscDbAPI(hostIP='genome-mysql.soe.ucsc.edu',db=db, username = username,pwd =pwd)



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

# There are 19804 protein coding genes we will be considering :)
# 19786 of these have unique transcriptional start sites.

# Get cre table data
creTable = getEncodeCreTable()

# Open the hg38 reference genome
with open("gencode/GRCh38.p10.genome.fa") as handle:
    genomeRecords = list(SeqIO.parse(handle, "fasta"))

# Collect all necessary data into a list of dictionaries
promoterDataDict = {}
cnt = 1
for i, tx in df.iterrows():
    # Test if cre is actually working
    print(cnt)
    chr = str(tx.seqname)
    acgtSeq = ''
    
    # chr-tssMinus2773-tssPlus50 is the key to the dictionary containing [txIds, seq, cres, chr, tss]
    if tx.strand == '+':
        tss = int(tx.start)
        tssPlus50 = int(tx.start) + 50
        tssMinus2773 = int(tx.start) - 2773
    elif tx.strand == '-':
        tss = int(tx.end)
        tssPlus50 = int(tx.end) - 50
        tssMinus2773 = int(tx.end) + 2773
        
    chrOneHot = chrToOneHot(chr)

    key = chr + "_" + str(tssMinus2773) + "_" + str(tssPlus50)
    print(key)
    if key in promoterDataDict: # We just need to add the new transcript id... The sequence for this tss was already written
        oldEntry = promoterDataDict[key]
        oldTxIds = oldEntry[0]
        newTxIds = oldTxIds + [str(tx.transcript_id)]
        promoterDataDict[key] = [newTxIds, oldEntry[1], oldEntry[2], oldEntry[3], oldEntry[4]]
    else:
        if tx.strand == '+':
            acgtSeq = getSeq(genomeRecords, chr, tssMinus2773, tssPlus50)
            creSeq = getCreSeq(creTable, chr, tssPlus50, tssMinus2773)
        elif tx.strand == '-':
            acgtSeq = reverseComplement(getSeq(genomeRecords, chr, tssPlus50, tssMinus2773))
            creSeq = reverseList(getCreSeq(creTable, chr, tssPlus50, tssMinus2773))
        acgtOneHot = seqToOneHot(acgtSeq)
        promoterDataDict[key] = [[tx.gene_id], acgtOneHot, creSeq, chrOneHot, tss]
    cnt += 1

print(len(promoterDataDict))

# Store list of dictionaries in promoterData_gene.jsonl.gz
with gzip.open('promoterData_gene.jsonl.gz', 'wb') as outfile:
    for key, val in promoterDataDict.items():
        writeStr = json.dumps({'id': key, 'tx_ids': val[0], 'acgtOneHot': val[1], 'creSeq': val[2], 'chrOneHot': val[3], 'tss': val[4]}) + '\n'
        outfile.write(writeStr.encode())
