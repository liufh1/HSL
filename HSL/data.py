import torch
import torch.utils.data as data
import os
import nltk
import pickle
import argparse


class TrainDataset(data.Dataset):
    """
    Load queries and products
    """

    def __init__(self, opt, vocab):
        self.vocab = vocab
        self.top_k = opt.top_k

        # Queries
        print('=> loading queries...')
        if not opt.use_dup:
            self.queries = pickle.load(open(os.path.join(opt.data_path, opt.data_name, 'train/train.queries.pkl'), 'rb'))
        else:
            self.queries = pickle.load(open(os.path.join(opt.data_path, opt.data_name, 'train/train.queries.duplicate.pkl'), 'rb'))
        print('=> queries loaded')

        # Products
        print('=> loading products...')
        self.prods = pickle.load(open(os.path.join(opt.data_path, opt.data_name, 'train/train.products.pkl'), 'rb'))
        print('=> products loaded...')

        # Load product object with image features
        print('=> loading product features...')
        self.prods.load_images(
            os.path.join(opt.data_path, opt.data_name, 'train/train.ids.txt'),
            os.path.join(opt.data_path, opt.data_name, 'train/train.features.npy'))
        print('=> product features loaded...')

        self.length = len(self.queries)

        print('=> train loader, loaded queries {}'.format(self.length))
        print('=> train loader, loaded products: {}'.format(len(self.prods)))

    def __getitem__(self, index):
        """
        Args:
            index: int; index of query

        Return:
            image: torch tensor of shape (2048); image feature of groundtruth product
            query: torch tensor of shape (?); query text id
            index: int; copy of input
        """
        # Query and groundtruth product pair
        query, gdprod = self.queries(index)
        # Load product image feature
        image = self.prods(gdprod, self.top_k)
        image = torch.Tensor(image)

        # Convert Query (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(query.lower())
        query = []
        query.append(self.vocab('<start>'))
        query.extend([self.vocab(token) for token in tokens])
        query.append(self.vocab('<end>'))
        query = torch.Tensor(query)

        return image, query, index

    def __len__(self):
        return self.length


class EvalDataset(data.Dataset):
    """
    Load queries and products
    """

    def __init__(self, opt, vocab, split):
        self.vocab = vocab
        self.top_k = opt.top_k

        # Queries
        print('=> loading queries...')
        self.queries = pickle.load(open(os.path.join(opt.data_path, opt.data_name, split, '{}.queries.pkl'.format(split)), 'rb'))
        print('=> queries loaded')

        # Products
        print('=> loading products...')
        self.prods = pickle.load(open(os.path.join(opt.data_path, opt.data_name, split, '{}.products.pkl'.format(split)), 'rb'))
        print('=> products loaded...')

        # Load product object with image features
        print('=> loading product features...')
        self.prods.load_images(
            os.path.join(opt.data_path, opt.data_name, '{}/{}.ids.txt'.format(split, split)),
            os.path.join(opt.data_path, opt.data_name, '{}/{}.features.npy'.format(split, split)))
        print('=> product features loaded...')

        self.length = len(self.queries)

        print('=> {} loader, loaded queries {}'.format(split, self.length))
        print('=> {} loader, loaded products: {}'.format(split, len(self.prods)))

    def __getitem__(self, index):
        """
        Args:
            index: int; index of query

        Return:
            images: torch tensor of shape (?, max, 2048); image features of candidate products
            prods: list of str of length (?); product ids of candidate products
            query: torch tensor of shape (?); query text
            query_id: str; id of query(used when loading validation set to look up answers)
            images.size(0): int; size of candidate pool
            index: int; copy of input
        """
        # Query and candidate pool
        (query_id, query), prods = self.queries(index)
        # Load candidate product image features
        images = [self.prods(prod, self.top_k) for prod in prods]
        num_boxes = [image.shape[0] for image in images]
        # （pool_size, max_box_num, feature_dim）
        padded_images = torch.zeros(len(images), max(num_boxes), images[0].shape[1])
        for i, image in enumerate(images):
            end = num_boxes[i]
            image = torch.Tensor(image)
            padded_images[i, :end] = image[:end]

        num_boxes = torch.Tensor(num_boxes).long()

        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(query.lower())
        query = []
        query.append(self.vocab('<start>'))
        query.extend([vocab(token) for token in tokens])
        query.append(self.vocab('<end>'))
        query = torch.Tensor(query)

        return padded_images, num_boxes, prods, query, query_id, index

    def __len__(self):
        return self.length


def collate_fn_train(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, query) tuple.
            - image: torch tensor of shape (num_boxes, 2048).
            - query: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 2048).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded query.
    """
    # Sort a data list by query length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, queries, ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    num_boxes = [image.size(0) for image in images]
    # batch_size, max_box_num, feature_dim
    padded_images = torch.zeros(len(images),max(num_boxes),images[0].size(1))
    for i, image in enumerate(images):
        end = num_boxes[i]
        padded_images[i,:end] = image[:end]

    num_boxes = torch.Tensor(num_boxes).long()


    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(query) for query in queries]
    targets = torch.zeros(len(queries), max(lengths)).long()
    for i, query in enumerate(queries):
        end = lengths[i]
        targets[i, :end] = query[:end]

    lengths = torch.Tensor(lengths).long()

    # return padded_images, num_boxes, targets, lengths, ids
    return padded_images, num_boxes, targets, lengths, ids


def collate_fn_eval(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (padded_images, num_boxes, prods, query, query_id, index) tuple.
            - padded_images: torch tensor of shape (?, 2048)
            - num_boxes: list of shape (?)
            - prods: str list of shape (?)
            - query: torch tensor of shape(?)
            - query_id: str
            - index: int

    Returns:
        images: torch tensor of shape (batch_size * ?, 2048).
        prods: list of candidate pools
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded query.
        query_ids: list of query_id
        pool_sizes: list of size of candidate pool
        ids: indices of queries
    """
    # Sort a data list by query length
    data.sort(key=lambda x: len(x[3]), reverse=True)
    imagess, num_boxess, prodss, queries, query_ids, ids = zip(*data)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    num_boxes = [images.size(1) for images in imagess]
    pool_sizes = [images.size(0) for images in imagess]
    # (batch_size, max_pool_size, max_boxes, feature_dim)
    padded_images = torch.zeros(len(imagess), max(pool_sizes), max(num_boxes), imagess[0].size(2))
    for i, images in enumerate(imagess):
        padded_images[i,:pool_sizes[i],:num_boxes[i]] = images


    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(query) for query in queries]
    targets = torch.zeros(len(queries), max(lengths)).long()
    for i, query in enumerate(queries):
        end = lengths[i]
        targets[i, :end] = query[:end]

    lengths = torch.Tensor(lengths).long()

    return padded_images, pool_sizes, num_boxess, prodss, targets, lengths, query_ids, ids


def get_train_loader(vocab, opt, shuffle=True):
    """
    Returns train set loader
    """
    trainset = TrainDataset(opt, vocab)

    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=opt.batch_size,
                                               shuffle=shuffle,
                                               pin_memory=True,
                                               num_workers=opt.num_workers,
                                               collate_fn=collate_fn_train)

    return train_loader


def get_test_loader(vocab, opt, split, batch_size=4):
    """
    Returns valid set loader or test set loader
    """
    evalset = EvalDataset(opt, vocab, split)

    test_loader = torch.utils.data.DataLoader(dataset=evalset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=opt.num_workers,
                                              collate_fn=collate_fn_eval)

    return test_loader