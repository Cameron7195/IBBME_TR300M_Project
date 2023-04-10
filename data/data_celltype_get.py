import json
import csv

'''
This file creates a dictionary object which maps GTEx sample IDs to their cell types, labelled
by string descriptions. For instance, one row might look like: "GTEX-1117F-0226-SM-5GZZ7": 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

The dictionary is saved to the file: sampleToCellType.json
'''

sample_to_celltype_dict = {}
sampleList = []

# Open the GTEx data to extract a list of sample IDs
with open("gtex/gene_tpm.gct") as infile:
    rows = csv.reader(infile, delimiter='\t')
    for row in rows:
        sampleList = row[2:]
        break

# Open a dictionary that relates cell type descriptions to their one-hot encodings
cellTypeEncodingDict = {}
with open("cellTypeEncoding.json", 'r') as infile:
    cellTypeEncodingDict = json.loads(infile.read())

# Open GTEx sample attributes and create dictionary
cnt = 0
with open("gtex/Annotations_SampleAttributes.txt") as infile:
    rowDicts = csv.DictReader(infile, delimiter='\t')
    for rowDict in rowDicts:
        sampleID = rowDict['SAMPID']
        cellTypeStr = rowDict['SMTSD']
        if sampleID in sampleList:
            sample_to_celltype_dict[sampleID] = cellTypeEncodingDict[cellTypeStr]

with open("sampleToCellType.json", 'w') as outfile:
    json.dump(sample_to_celltype_dict, outfile)
