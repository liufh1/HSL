import argparse


def get_parser():
	# Main parser args
    parser = argparse.ArgumentParser(description='Hierarchical-Similarity-Learning-for-Language-based-Product-Image-Retrieval')

    # Basics for training
    parser.add_argument('--data_path', default='../data',
                        help='path to datasets')
    parser.add_argument('--data_name', default='KDDCup2020',
                        help='denote datasets')
    parser.add_argument('--use_dup', action='store_true',
                        help='Use duplicate training set.')
    parser.add_argument('--vocab_path', default='../data/KDDCup2020',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--decay_rate', default=0.9, type=float,
                        help='Learning rate decay rate.')
    parser.add_argument('--step_size', default=1, type=int,
                        help='Step size to decay learning rate.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--log_step', default=50, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default=None,
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none)')
    parser.add_argument('--test', action='store_true',
                        help='Only evaluate on valid set.')
    parser.add_argument('--submit', default=None, choices=['testA', 'testB'],
                        help='Generate submission file.')
    parser.add_argument('--out_path', default=None, type=str,
                        help='Path to save submission file')
    parser.add_argument('--score_to', default=None, type=str,
                        help='Path to save scores file')

    # Embedding Related
    parser.add_argument('--word_dim', default=512, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--glove', action='store_true',
                        help='Use GloVe embedding.')

    #subparsers = parser.add_subparsers(help='subparser help')

    # Add model specific args here by adding subparsers
    # self-attention
    #parser = subparsers.add_parser('selfattn', help='selfattn help')
    parser.add_argument('--base_loss', default='ContrastiveLoss', type=str,
                        help='Basic loss function.')
    parser.add_argument('--top_k', default=20, type=int,
                        help='Top k bounding boxes in terms of area.')
    parser.set_defaults(model='selfattn')
    parser.add_argument('--img_num_layers', default=6, type=int,
                        help='Number of image encoder layers.')
    parser.add_argument('--txt_num_layers', default=4, type=int,
                        help='Number of text encoder layers.')
    parser.add_argument('--d_model', default=512, type=int,
                        help='Dimensionality of transformer hidden layers.')
    parser.add_argument('--score_agg', default='max', type=str,
                        help='Style to aggregate scores of bboxes.')

    parser.add_argument('--HIDDEN_SIZE', default=512, type=int,
                        help='')
    parser.add_argument('--MULTI_HEAD', default=8, type=int,
                        help='')
    parser.add_argument('--DROPOUT_R', default=0.1, type=float,
                        help='')
    parser.add_argument('--FLAT_MLP_SIZE', default=512, type=int,
                        help='')
    parser.add_argument('--FLAT_GLIMPSES', default=1, type=int,
                        help='')
    parser.add_argument('--FLAT_OUT_SIZE', default=1024, type=int,
                        help='')

    # parser.print_help()
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
	get_parser()