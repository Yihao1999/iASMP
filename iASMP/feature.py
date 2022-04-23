from collections import Counter

from Bio import SeqIO

import numpy as np

import warnings
import math

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
#from gensim.models import Word2Vec

MAX_length = 130

def add(x, i):
    x_copy = x.copy()
    x_copy[i] = 1
    return x_copy




def BLOSUM62(seq):
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    }

    pad_len = MAX_length - len(seq)
    seqs = []
    for aa in seq:
        seqs.append(blosum62[aa])
    for _ in range(pad_len):
        seqs.append(blosum62['-'])

    return seqs


def Count(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * MAX_length)
                    break
        if myCount == 0:
            code.append(0)
    return code


def CTDD(seq):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []

    code = []
    for p in property:
        code = code + Count(group1[p], seq) + Count(group2[p], seq) + Count(group3[p], seq)
    encodings.append(code)
    return encodings


def DPC(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    # header = ['#'] + diPeptides
    # encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    # for i in fastas:
    # name, sequence = i[0], re.sub('-', '', i[1])
    code = []
    tmpCode = [0] * 400
    for j in range(len(seq) - 2 + 1):
        tmpCode[AADict[seq[j]] * 20 + AADict[seq[j + 1]]] = tmpCode[AADict[seq[j]] * 20 + AADict[
            seq[j + 1]]] + 1
    if sum(tmpCode) != 0:
        tmpCode = [i / sum(tmpCode) for i in tmpCode]
    code = code + tmpCode
    encodings.append(code)
    return encodings


def AAC(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    # AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []

    # for i in fastas:
    # name, sequence = i[0], re.sub('-', '', i[1])
    count = Counter(seq)
    for key in count:
        count[key] = count[key] / len(seq)
    code = []
    for aa in AA:
        code.append(count[aa])
    encodings.append(code)
    return encodings


def ZSCALE(seq):
    zscale = {
        'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
        'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
        'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
        'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
        'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
        'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
        'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
        'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
        'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
        'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
        'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
        'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
        'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
        'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
        'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
        'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
        'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
        'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
        'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
        'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
        '-': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
    }
    encodings = []
    # header = ['#']
    # for p in range(1, len(fastas[0][1]) + 1):
    #     for z in ('1', '2', '3', '4', '5'):
    #         header.append('Pos' + str(p) + '.ZSCALE' + z)
    # encodings.append(header)

    # for i in fastas:
    # name, sequence = i[0], i[1]
    code = []

    for _ in range(MAX_length - len(seq)):
        code = code + zscale['-']

    for aa in seq:
        code = code + zscale[aa]
    encodings.append(code)
    return encodings


def TPC(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    # for i in fastas:
    # name, sequence = i[0], re.sub('-', '', i[1])
    code = []
    tmpCode = [0] * 8000
    for j in range(len(seq) - 3 + 1):
        tmpCode[AADict[seq[j]] * 400 + AADict[seq[j + 1]] * 20 + AADict[seq[j + 2]]] = tmpCode[AADict[seq[j]] * 400 +
                                                                                               AADict[seq[j + 1]] * 20 +
                                                                                               AADict[seq[j + 2]]] + 1
    if sum(tmpCode) != 0:
        tmpCode = [i / sum(tmpCode) for i in tmpCode]
    code = code + tmpCode
    encodings.append(code)
    return encodings


def DDE(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'

    myCodons = {
        'A': 4,
        'C': 2,
        'D': 2,
        'E': 2,
        'F': 2,
        'G': 4,
        'H': 2,
        'I': 3,
        'K': 2,
        'L': 6,
        'M': 1,
        'N': 2,
        'P': 4,
        'Q': 2,
        'R': 6,
        'S': 6,
        'T': 4,
        'V': 4,
        'W': 1,
        'Y': 2
    }

    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    # for i in fastas:
    # name, sequence = i[0], re.sub('-', '', i[1])
    code = []
    tmpCode = [0] * 400
    for j in range(len(seq) - 2 + 1):
        tmpCode[AADict[seq[j]] * 20 + AADict[seq[j + 1]]] = tmpCode[AADict[seq[j]] * 20 + AADict[
            seq[j + 1]]] + 1
    if sum(tmpCode) != 0:
        tmpCode = [i / sum(tmpCode) for i in tmpCode]

    myTV = []
    for j in range(len(myTM)):
        myTV.append(myTM[j] * (1 - myTM[j]) / (len(seq) - 1))

    for j in range(len(tmpCode)):
        tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

    code = code + tmpCode
    encodings.append(code)
    return encodings


def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + gap + 1 < len(sequence) and i + 2 * gap + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i + gap + 1]] + '.' + AADict[
                    sequence[i + 2 * gap + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res


def CTriad(seq):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []
    # header = ['#']
    # for f in features:
    #     header.append(f)
    # encodings.append(header)

    # me, sequence = i[0], re.sub('-', '', i[1])
    code = []
    if len(seq) < 3:
        print('Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n')
        return 0
    code = code + CalculateKSCTriad(seq, 0, features, AADict)
    encodings.append(code)

    return encodings


def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i + g + 1]] + '.' + AADict[sequence[i + 2 * g + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res


def KSCTriad(seq, gap=1):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    encodings = []

    code = []
    if len(seq) < 2 * gap + 3:
        print('Error: for "KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3). \n\n')
        return 0
    code = code + CalculateKSCTriad(seq, gap, features, AADict)
    encodings.append(code)

    return encodings


def GTPC(seq):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()
    baseNum = len(groupKey)
    triple = [g1 + '.' + g2 + '.' + g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    encodings = []

    code = []
    myDict = {}
    for t in triple:
        myDict[t] = 0

    sum = 0
    for j in range(len(seq) - 3 + 1):
        myDict[index[seq[j]] + '.' + index[seq[j + 1]] + '.' + index[seq[j + 2]]] = myDict[index[seq[j]] + '.' + index[
            seq[j + 1]] + '.' + index[seq[j + 2]]] + 1
        sum = sum + 1

    if sum == 0:
        for t in triple:
            code.append(0)
    else:
        for t in triple:
            code.append(myDict[t] / sum)
    encodings.append(code)

    return encodings


def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1 + '.' + key2] = 0
    return gPair


def CKSAAGP(seq, gap=2):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    encodings = []

    code = []
    for g in range(gap + 1):
        gPair = generateGroupPairs(groupKey)
        sum = 0
        for p1 in range(len(seq)):
            p2 = p1 + g + 1
            if p2 < len(seq) and seq[p1] in AA and seq[p2] in AA:
                gPair[index[seq[p1]] + '.' + index[seq[p2]]] = gPair[index[seq[p1]] + '.' + index[
                    seq[p2]]] + 1
                sum = sum + 1

        if sum == 0:
            for gp in gPairIndex:
                code.append(0)
        else:
            for gp in gPairIndex:
                code.append(gPair[gp] / sum)

    encodings.append(code)

    return encodings


def GAAC(seq):
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    groupKey = group.keys()

    encodings = []
    code = []
    count = Counter(seq)
    myDict = {}
    for key in groupKey:
        for aa in group[key]:
            myDict[key] = myDict.get(key, 0) + count[aa]

    for key in groupKey:
        code.append(myDict[key] / len(seq))
    encodings.append(code)

    return encodings


def GDPC(seq):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()
    baseNum = len(groupKey)
    dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    encodings = []

    code = []
    myDict = {}
    for t in dipeptide:
        myDict[t] = 0

    sum = 0
    for j in range(len(seq) - 2 + 1):
        myDict[index[seq[j]] + '.' + index[seq[j + 1]]] = myDict[index[seq[j]] + '.' + index[
            seq[j + 1]]] + 1
        sum = sum + 1

    if sum == 0:
        for t in dipeptide:
            code.append(0)
    else:
        for t in dipeptide:
            code.append(myDict[t] / sum)
    encodings.append(code)

    return encodings


def AAINDEX(seq):
    temp = "-" * (MAX_length - len(seq))
    seq += temp

    AA = 'ARNDCQEGHILKMFPSTWYV'

    fileAAindex = "data\\AAindex1.txt"
    with open(fileAAindex) as f:
        records = f.readlines()[1:]

    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    encodings = []

    code = []

    for aa in seq:
        if aa == '-':
            for j in AAindex:
                code.append(0)

            continue
        for j in AAindex:
            code.append(j[index[aa]])

    encodings.append(code)

    return encodings


def CTDT(seq):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []

    code = []
    aaPair = [seq[j:j + 2] for j in range(len(seq) - 1)]
    for p in property:
        c1221, c1331, c2332 = 0, 0, 0
        for pair in aaPair:
            if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                c1221 = c1221 + 1
                continue
            if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                c1331 = c1331 + 1
                continue
            if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                c2332 = c2332 + 1
        code = code + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
    encodings.append(code)
    return encodings


def Geary(seq, props=['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102',
                      'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
          nlag=2):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    fileAAidx = "data\\AAidx.txt"
    with open(fileAAidx) as f:
        records = f.readlines()[1:]
    myDict = {}
    for i in records:
        array = i.rstrip().split('\t')
        myDict[array[0]] = array[1:]

    AAidx = []
    AAidxName = []
    for i in props:
        if i in myDict:
            AAidx.append(myDict[i])
            AAidxName.append(i)
        else:
            print('"' + i + '" properties not exist.')
            return None

    AAidx1 = np.array([float(j) for i in AAidx for j in i])
    AAidx = AAidx1.reshape((len(AAidx), 20))

    propMean = np.mean(AAidx, axis=1)
    propStd = np.std(AAidx, axis=1)

    for i in range(len(AAidx)):
        for j in range(len(AAidx[i])):
            AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    encodings = []

    code = []
    N = len(seq)
    for prop in range(len(props)):
        xmean = sum([AAidx[prop][index[aa]] for aa in seq]) / N
        for n in range(1, nlag + 1):
            if len(seq) > nlag:
                # if key is '-', then the value is 0
                rn = (N - 1) / (2 * (N - n)) * ((sum(
                    [(AAidx[prop][index.get(seq[j], 0)] - AAidx[prop][index.get(seq[j + n], 0)]) ** 2 for
                     j in range(len(seq) - n)])) / (sum(
                    [(AAidx[prop][index.get(seq[j], 0)] - xmean) ** 2 for j in range(len(seq))])))
            else:
                rn = 'NA'
            code.append(rn)
    encodings.append(code)
    return encodings


def CKSAAP(seq, gap=2, **kw):

    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    code = []
    for g in range(gap + 1):
        myDict = {}
        for pair in aaPairs:
            myDict[pair] = 0
        sum = 0
        for index1 in range(len(seq)):
            index2 = index1 + g + 1
            if index1 < len(seq) and index2 < len(seq) and seq[index1] in AA and seq[
                index2] in AA:
                myDict[seq[index1] + seq[index2]] = myDict[seq[index1] + seq[index2]] + 1
                sum = sum + 1
        for pair in aaPairs:
            code.append(myDict[pair] / sum)
    encodings.append(code)
    return encodings


def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def PAAC(seq, lambdaValue=3, w=0.05):


    dataFile = 'data\PAAC.txt'
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []



    code = []
    theta = []
    for n in range(1, lambdaValue + 1):
        theta.append(
            sum([Rvalue(seq[j], seq[j + n], AADict, AAProperty1) for j in range(len(seq) - n)]) / (
                    len(seq) - n))
    myDict = {}
    for aa in AA:
        myDict[aa] = seq.count(aa)
    code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
    code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
    encodings.append(code)
    return encodings

# AFC-T, AFC-CP

def Feature(f):
    amino_acids = "XACDEFGHIKLMNPQRSTVWY"
    amino_acids_dict = {}
    seqs = []
    seqs_blosum62 = []
    seqs_dde = []
    seqs_z = []
    seqs_dpc = []
    seqs_aac = []
    seqs_ctdd = []
    lable_seqs = []
    work2vec = []
    seqs_sr = []
    seqs_ksctriad = []
    seqs_gtpc = []
    seqs_cksaagp = []
    seqs_gaac = []
    seqs_gdpc = []
    seqs_aaindex = []
    seqs_ctdt = []
    seqs_geary = []
    seqs_cksaap = []
    seqs_ctrial = []
    seqs_paac = []
    for n, s in enumerate(amino_acids):
        amino_acids_dict[s] = n
    #new_antifu = Word2Vec.load('fa_model_All.bin')

    for n, s in enumerate(SeqIO.parse(f, "fasta")):
        seq_blosum62 = BLOSUM62(s.seq)
        seq_ksctriad = KSCTriad(s.seq)
        seq_dde = DDE(s.seq)
        seq_z = ZSCALE(s.seq)
        seq_aac = AAC(s.seq)
        seq_dpc = DPC(s.seq)
        seq_ctdd = CTDD(s.seq)
        seq_ctrial = CTriad(s.seq)
        seq_gtpc = GTPC(s.seq)
        seq_cksaagp = CKSAAGP(s.seq)
        seq_gaac = GAAC(s.seq)
        seq_gdpc = GDPC(s.seq)
        seq_ctdt = CTDT(s.seq)
        seq_geary = Geary(s.seq)
        seq_cksaap = CKSAAP(s.seq)
        seq_aaindex = AAINDEX(s.seq)
        seq_paac = PAAC(s.seq)
        seqs_dde.append(seq_dde)
        seqs_z.append(seq_z)
        seqs_aac.append(seq_aac)
        seqs_dpc.append(seq_dpc)
        seqs_ctdd.append(seq_ctdd)
        seqs_blosum62.append(seq_blosum62)
        seqs_ctrial.append(seq_ctrial)
        seqs_ksctriad.append(seq_ksctriad)
        seqs_gtpc.append(seq_gtpc)
        seqs_cksaagp.append(seq_cksaagp)
        seqs_gaac.append(seq_gaac)
        seqs_gdpc.append(seq_gdpc)
        seqs_ctdt.append(seq_ctdt)
        seqs_geary.append(seq_geary)
        seqs_cksaap.append(seq_cksaap)
        seqs_aaindex.append(seq_aaindex)
        seqs_paac.append(seq_paac)
        temp_pad = []
        temp_pad1 = []
        temps = []
        for i in range(20):
            temp_pad1.append(0)
        for i in range(MAX_length - len(s)):
            temps.append(temp_pad1)
        for i in range(MAX_length - len(str(s.seq))):
            temp_pad.append(0)
        train_seq = [amino_acids_dict[a.upper()] for a in str(s.seq).upper()] + temp_pad

        seqs_sr.append(train_seq)
        #aux_p3 = [new_antifu.wv[a] if a in "ACDEFGHIKLMNPQRSTVWY" else [0 for i in range(20)] for a in
                  #str(s.seq).upper()] + temps
        #work2vec.append(aux_p3)
        if s.id[-1] == "1":

            lable_seqs.append([1])
        else:
            lable_seqs.append([0])

    return seqs_blosum62, lable_seqs, work2vec, seqs_sr, seqs_dde, seqs_z, seqs_aac, seqs_dpc, seqs_ctdd, seqs_ctrial, seqs_ksctriad, seqs_gtpc, seqs_cksaagp, seqs_gaac, seqs_gdpc, seqs_ctdt, seqs_geary, seqs_cksaap, seqs_aaindex, seqs_paac

# AFC-C based on main dataset

def Feature1(f):
    amino_acids = "XACDEFGHIKLMNPQRSTVWY"
    amino_acids_dict = {}
    seqs = []
    seqs_blosum62 = []
    seqs_dde = []
    seqs_z = []
    seqs_dpc = []
    seqs_aac = []
    seqs_ctdd = []
    lable_seqs = []
    work2vec = []
    seqs_sr = []
    seqs_ksctriad = []
    seqs_gtpc = []
    seqs_cksaagp = []
    seqs_gaac = []
    seqs_gdpc = []
    seqs_aaindex = []
    seqs_ctdt = []
    seqs_geary = []
    seqs_cksaap = []
    seqs_ctrial = []
    seqs_paac = []
    for n, s in enumerate(amino_acids):
        amino_acids_dict[s] = n
    #new_antifu = Word2Vec.load('D:\E下载\Dataset\Dataset\\fa_model_All.bin')

    for n, s in enumerate(SeqIO.parse(f, "fasta")):
        seq_blosum62 = BLOSUM62(s.seq)
        #seq_ksctriad = KSCTriad(s.seq)
        seq_dde = DDE(s.seq)
        seq_z = ZSCALE(s.seq)
        seq_aac = AAC(s.seq)
        seq_dpc = DPC(s.seq)
        seq_ctdd = CTDD(s.seq)
        #seq_ctrial = CTriad(s.seq)
        seq_gtpc = GTPC(s.seq)
        seq_cksaagp = CKSAAGP(s.seq)
        seq_gaac = GAAC(s.seq)
        seq_gdpc = GDPC(s.seq)
        seq_ctdt = CTDT(s.seq)
        seq_geary = Geary(s.seq)
        #seq_cksaap = CKSAAP(s.seq)

        seq_aaindex = AAINDEX(s.seq)
        #seq_paac = PAAC(s.seq)

        seqs_dde.append(seq_dde)
        seqs_z.append(seq_z)
        seqs_aac.append(seq_aac)
        seqs_dpc.append(seq_dpc)
        seqs_ctdd.append(seq_ctdd)
        seqs_blosum62.append(seq_blosum62)
        #seqs_ctrial.append(seq_ctrial)
        #seqs_ksctriad.append(seq_ksctriad)
        seqs_gtpc.append(seq_gtpc)
        seqs_cksaagp.append(seq_cksaagp)
        seqs_gaac.append(seq_gaac)
        seqs_gdpc.append(seq_gdpc)
        seqs_ctdt.append(seq_ctdt)
        seqs_geary.append(seq_geary)
        #seqs_cksaap.append(seq_cksaap)
        seqs_aaindex.append(seq_aaindex)
        #seqs_paac.append(seq_paac)
        temp_pad = []
        temp_pad1 = []
        temps = []
        for i in range(20):
            temp_pad1.append(0)
        for i in range(MAX_length - len(s)):
            temps.append(temp_pad1)
        for i in range(MAX_length - len(str(s.seq))):
            temp_pad.append(0)
        train_seq = [amino_acids_dict[a.upper()] for a in str(s.seq).upper()] + temp_pad

        seqs_sr.append(train_seq)
        #aux_p3 = [new_antifu.wv[a] if a in "ACDEFGHIKLMNPQRSTVWY" else [0 for i in range(20)] for a in
                  #str(s.seq).upper()] + temps
        #work2vec.append(aux_p3)
        if s.id[-1] == "1":
            lable_seqs.append([1])
        else:
            lable_seqs.append([0])

    return seqs_blosum62, lable_seqs, work2vec, seqs_sr, seqs_dde, seqs_z, seqs_aac, seqs_dpc, seqs_ctdd, seqs_ctrial, seqs_ksctriad, seqs_gtpc, seqs_cksaagp, seqs_gaac, seqs_gdpc, seqs_ctdt, seqs_geary, seqs_cksaap, seqs_aaindex, seqs_paac

# AFC-C based on alternate dataset

def Feature2(f):
    amino_acids = "XACDEFGHIKLMNPQRSTVWY"
    amino_acids_dict = {}
    seqs = []
    seqs_blosum62 = []
    seqs_dde = []
    seqs_z = []
    seqs_dpc = []
    seqs_aac = []
    seqs_ctdd = []
    lable_seqs = []
    work2vec = []
    seqs_sr = []
    seqs_ksctriad = []
    seqs_gtpc = []
    seqs_cksaagp = []
    seqs_gaac = []
    seqs_gdpc = []
    seqs_aaindex = []
    seqs_ctdt = []
    seqs_geary = []
    seqs_cksaap = []
    seqs_ctrial = []
    seqs_paac = []
    for n, s in enumerate(amino_acids):
        amino_acids_dict[s] = n
    #new_antifu = Word2Vec.load('D:\E下载\Dataset\Dataset\\fa_model_All.bin')

    for n, s in enumerate(SeqIO.parse(f, "fasta")):
        seq_blosum62 = BLOSUM62(s.seq)
        #seq_ksctriad = KSCTriad(s.seq)
        seq_dde = DDE(s.seq)
        seq_z = ZSCALE(s.seq)
        seq_aac = AAC(s.seq)
        seq_dpc = DPC(s.seq)
        seq_ctdd = CTDD(s.seq)
        seq_ctrial = CTriad(s.seq)
        seq_gtpc = GTPC(s.seq)
        seq_cksaagp = CKSAAGP(s.seq)
        seq_gaac = GAAC(s.seq)
        seq_gdpc = GDPC(s.seq)
        seq_ctdt = CTDT(s.seq)
        seq_geary = Geary(s.seq)
        #seq_cksaap = CKSAAP(s.seq)

        seq_aaindex = AAINDEX(s.seq)
        #seq_paac = PAAC(s.seq)

        seqs_dde.append(seq_dde)
        seqs_z.append(seq_z)
        seqs_aac.append(seq_aac)
        seqs_dpc.append(seq_dpc)
        seqs_ctdd.append(seq_ctdd)
        seqs_blosum62.append(seq_blosum62)
        seqs_ctrial.append(seq_ctrial)
        #seqs_ksctriad.append(seq_ksctriad)
        seqs_gtpc.append(seq_gtpc)
        seqs_cksaagp.append(seq_cksaagp)
        seqs_gaac.append(seq_gaac)
        seqs_gdpc.append(seq_gdpc)
        seqs_ctdt.append(seq_ctdt)
        seqs_geary.append(seq_geary)
        #seqs_cksaap.append(seq_cksaap)
        seqs_aaindex.append(seq_aaindex)
        #seqs_paac.append(seq_paac)
        temp_pad = []
        temp_pad1 = []
        temps = []
        for i in range(20):
            temp_pad1.append(0)
        for i in range(MAX_length - len(s)):
            temps.append(temp_pad1)
        for i in range(MAX_length - len(str(s.seq))):
            temp_pad.append(0)
        train_seq = [amino_acids_dict[a.upper()] for a in str(s.seq).upper()] + temp_pad

        seqs_sr.append(train_seq)
        #aux_p3 = [new_antifu.wv[a] if a in "ACDEFGHIKLMNPQRSTVWY" else [0 for i in range(20)] for a in
                  #str(s.seq).upper()] + temps
        #work2vec.append(aux_p3)
        if s.id[-1] == "1":
            lable_seqs.append([1])
        else:
            lable_seqs.append([0])

    return seqs_blosum62, lable_seqs, work2vec, seqs_sr, seqs_dde, seqs_z, seqs_aac, seqs_dpc, seqs_ctdd, seqs_ctrial, seqs_ksctriad, seqs_gtpc, seqs_cksaagp, seqs_gaac, seqs_gdpc, seqs_ctdt, seqs_geary, seqs_cksaap, seqs_aaindex, seqs_paac





