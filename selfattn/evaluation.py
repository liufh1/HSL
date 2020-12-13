import os
import zipfile
import pandas as pd
import time
import numpy as np
import torch
from utils import *
from .loss import *
from data import Scores


def evaluate(model, data_loader, scorer, answer, log_step=50, logging=print, out_path=None):
    """
        Evaluate with nDCG scorer
    """
    batch_time = AverageMeter()
    ndcgs = AverageMeter()

    # save scores
    if out_path is not None:
        record = Scores()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    for index, (padded_images, pool_sizes, num_boxess, prodss, queries, lengths, query_ids, _) in enumerate(data_loader):
        # calculate nDCG
        for i, size in enumerate(pool_sizes):
            imgs = padded_images[i,:size,:,:]
            query = queries[i].view(1, -1).expand(imgs.size(0), -1)
            lens = lengths[i].view(1).expand(imgs.size(0))
            img_embs, query_embs, img_embs2, query_embs2, img_embs3, query_embs3, img_embs4, query_embs4 = model.forward_emb(imgs, num_boxess[i], query, lens, no_grad=True)
            # a query <=> candidate pool
            # calculate similarity scores between query and candidate pool
            # => pool_size, max_boxes
            scores1 = model.cal_scores(img_embs, query_embs, num_boxess[i])[0]
            scores2 = model.cal_scores(img_embs2, query_embs2, num_boxess[i])[0]
            scores3 = model.cal_scores2(img_embs3, query_embs3, num_boxess[i])[0]
            scores4 = model.cal_scores2(img_embs4, query_embs4, num_boxess[i])[0]
            scores1 = scores1.cpu().data.numpy()
            scores2 = scores2.cpu().data.numpy()
            scores3 = scores3.cpu().data.numpy()
            scores4 = scores4.cpu().data.numpy()
            scores =  scores1 + 0.5 * scores2 + scores3 + scores4 * 0.5

            # save scores
            if out_path is not None:
                for j, score in enumerate(scores):
                    record.add_item(query_ids[i], prodss[i][j], score.item())

            # calculate nDCG
            ndcg = cal_ndcg(scores, prodss[i], answer[query_ids[i]], scorer)
            ndcgs.update(ndcg)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % log_step == 0:
            logging(
                'Eval: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(index, len(data_loader), batch_time=batch_time))

    if out_path is not None:
        record.dump(out_path)
        logging('Scores dumped to {}'.format(out_path))

    return ndcgs.avg



def cal_ndcg(sim, cand_pool, gdprods, scorer):
    # sort in descending order
    index = np.argsort(sim)[::-1]

    sorted_labels = []
    for i in range(index.shape[0]):
        # top i product is in groundtruth product pools
        if int(cand_pool[index[i]]) in gdprods:
            sorted_labels.append(1)
        else:
            sorted_labels.append(0)

    return scorer.score(sorted_labels)
