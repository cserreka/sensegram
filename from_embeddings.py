import argparse
from multiprocessing import cpu_count
from os.path import join
from time import time
import filter_clusters
import vector_representations.build_sense_vectors
from utils.common import ensure_dir
from utils.common import exists
from word_graph import compute_graph_of_related_words
from word_sense_induction import ego_network_clustering


def word_sense_induction(neighbours_fpath, clusters_fpath, n, threads):
    print("\nStart clustering of word ego-networks.")
    tic = time()
    ego_network_clustering(neighbours_fpath, clusters_fpath, max_related=n, num_cores=threads)
    print("Elapsed: {:f} sec.".format(time() - tic))


def building_sense_embeddings(clusters_minsize_fpath, vectors_fpath):
    print("\nStart pooling of word vectors.")
    vector_representations.build_sense_vectors.run(
        clusters_minsize_fpath, vectors_fpath, sparse=False,
        norm_type="sum", weight_type="score", max_cluster_words=20)


def main():
    parser = argparse.ArgumentParser(description='Performs training of a word sense embeddings model from a raw text '
                                                 'corpus using the SkipGram approach based on word2vec and graph '
                                                 'clustering of ego networks of semantically related terms.')

    parser.add_argument('-vectors', help="Existing embeddings to make sense vectors from")
    parser.add_argument('-threads', help="Use <int> threads (default {}).".format(cpu_count()),
                        default=cpu_count(), type=int)
    parser.add_argument('-iter', help="Run <int> training iterations (default 5).", default=5, type=int)
    parser.add_argument('-min_count', help="This will discard words that appear less than <int> times"
                                           " (default is 10).", default=10, type=int)
    parser.add_argument('-N', help="Number of nodes in each ego-network (default is 200).", default=200, type=int)
    parser.add_argument('-n', help="Maximum number of edges a node can have in the network"
                                   " (default is 200).", default=200, type=int)
    parser.add_argument('-min_size', help="Minimum size of the cluster (default is 5).", default=5, type=int)

    args = parser.parse_args()

    model_dir = "model/"
    ensure_dir(model_dir)
    vectors_fpath = args.vectors
    neighbours_fpath = join(model_dir, args.vectors + ".N{}.graph".format(args.N))
    clusters_fpath = join(model_dir, args.vectors + ".n{}.clusters".format(args.n))
    clusters_minsize_fpath = clusters_fpath + ".minsize" + str(args.min_size)  # clusters that satisfy min_size
    clusters_removed_fpath = clusters_minsize_fpath + ".removed"  # cluster that are smaller than min_size

    if exists(vectors_fpath):
        print("Using existing vectors:", vectors_fpath)
    else:
        return FileNotFoundError

    if not exists(neighbours_fpath):
        compute_graph_of_related_words(vectors_fpath, neighbours_fpath, neighbors=args.N)
    else:
        print("Using existing neighbors:", neighbours_fpath)

    word_sense_induction(neighbours_fpath, clusters_fpath, args.n, args.threads)
    filter_clusters.run(clusters_fpath, clusters_minsize_fpath, args.min_size)
    building_sense_embeddings(clusters_minsize_fpath, vectors_fpath)


if __name__ == '__main__':
    main()
