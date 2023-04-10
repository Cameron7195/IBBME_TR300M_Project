import json
import csv
import numpy as np
import gzip

'''
This file creates the primary expression array that the model will predict.
This array is sized 19786 x 17382. It contains the expression levels for 19786 genes,
each measured in 17382 samples.
'''

cnt = 0



# There are 19786 rows in promoterData (19786 unique promoter+enhancers, with related info)
expressionTempArray = np.zeros((19786, 17382))

# We must relate each item of promoterData to row(s) of the gene_tpm table.
# There are 56200 rows in the gene_tpm table. For each promoterData then,
# we must essentially search through this entire table to find the gene(s) that
# correspond and add them to the expression array. However, the time complexity for
# this operation is prohibitive, and it is much faster to store as many rows of gene_tpm
# in a dictionary with the gene id as the key, therefore, instead of searching through
# gene_tpm, we can simply do a hash read in constant time.

# We can store at most around 30000 rows of gene_tpm in a nested dict before my computer
# runs out of memory. Doing this 2 times is way faster than searching through the
# gene_tpm elements.

for i in range(2):
    txSampleExpress_dict = {}
    # with open("gtex/transcript_tpm.gct") as infile:
    with open("gtex/gene_tpm.gct") as infile:
        txExpressDicts = csv.DictReader(infile, delimiter='\t')
        
        cnt = 0
        for txD in txExpressDicts:
            if cnt < (30000*i) or cnt >= (30000*(i+1)):
                #print("Not counting: " + str(cnt))
                cnt += 1
                continue
            # tx_id = txD['transcript_id']
            tx_id = txD['Name']
            txSampleExpress_dict[tx_id] = txD
            if cnt % 100 == 0:
                print(cnt)
            cnt += 1
        print(cnt)
    
    tempArrayIdx0 = 0
    with gzip.open('promoterData_gene.jsonl.gz', 'rt', encoding='UTF-8') as infile:
        for line in infile:
            lineDict = json.loads(line)
            tx_ids = lineDict['tx_ids']
            #print(tx_ids)
            for id in tx_ids:
                if id in txSampleExpress_dict:
                    tempArrayIdx1 = 0
                    for column, val in txSampleExpress_dict[id].items():
                        if column != 'Name' and column != 'Description':
                            expressionTempArray[tempArrayIdx0, tempArrayIdx1] += float(val)
                            tempArrayIdx1 += 1
            tempArrayIdx0 += 1
    del txExpressDicts

expressionArr = np.load(gzip.GzipFile('expressionArr_gene.npy.gz', 'r'))

f = gzip.GzipFile("expressionArr_gene.npy.gz", "w")
np.save(file=f, arr=expressionTempArray)
f.close()