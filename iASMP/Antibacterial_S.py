from sklearn.ensemble import ExtraTreesClassifier
from feature import Feature
import numpy as np
from sklearn.model_selection import cross_val_score


def init_data(file_name):
    seqs_blosum62, label, work2vec, seqs_sr, seqs_dde, seqs_z, seqs_aac, seqs_dpc, seqs_ctdd, seqs_ctrial, seqs_ksctriad, seqs_gtpc, seqs_cksaagp, seqs_gaac, seqs_gdpc, seqs_ctdt, seqs_geary, seqs_cksaap, seqs_aaindex, seqs_paac = Feature(
        file_name)
    seqs_gaac = np.array(seqs_gaac)
    seqs_gaac = seqs_gaac.reshape(seqs_gaac.shape[0], -1)
    seqs_paac = np.array(seqs_paac)
    seqs_paac = seqs_paac.reshape(seqs_paac.shape[0], -1)
    seqs_ctdt = np.array(seqs_ctdt)
    seqs_ctdt = seqs_ctdt.reshape(seqs_ctdt.shape[0], -1)
    seqs_gtpc = np.array(seqs_gtpc)
    seqs_gtpc = seqs_gtpc.reshape(seqs_gtpc.shape[0], -1)
    seqs_geary = np.array(seqs_geary)
    seqs_geary = seqs_geary.reshape(seqs_geary.shape[0], -1)
    seqs_gaac = seqs_gaac.reshape(seqs_gaac.shape[0], -1)
    data_features1 = np.concatenate((seqs_paac, seqs_ctdt, seqs_gaac, seqs_gtpc, seqs_geary), 1)
    label = np.array(label)
    label = label.reshape(label.shape[0], )
    return data_features1, label


if __name__ == "__main__":
    clf = ExtraTreesClassifier(random_state=0)

    data_features, label = init_data("dataset_S.txt")

    part_data_features = data_features[:,
                         [4, 133, 11, 13, 147, 150, 151, 158, 161, 163, 40, 170, 172, 174, 175, 49, 56, 64, 65, 66, 71,
                          206, 84, 85, 93, 97, 106, 107, 108, 109, 123, 125, 126]]

    print(cross_val_score(clf, part_data_features, label, cv=10).mean())

