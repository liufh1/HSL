#***************************************************************************
#Hierarchical-Similarity-Learning-for-Language-based-Product-Image-Retrieval
#***************************************************************************

import pickle
import os
import sys
import json
import time
import shutil

import torch

import importlib
from vocab import Vocabulary
from data import Query, Product
from utils import AverageMeter, get_we_parameter
from metric import NDCGScorer

import logging
import tensorboard_logger as tb_logger

import parser

def main():

    # parse command line args
    opt = parser.get_parser()
    

    
    # config logger
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    if not opt.test and not opt.submit:
        tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = pickle.load(open('../data/KDDCup2020/kddcup_vocab.pkl', 'rb'))
    opt.vocab_size = len(vocab)
    logging.info('vocabulary size: {}'.format(opt.vocab_size))

    if opt.glove:
        print("loading glove word2vec......")
        opt.we_parameter = get_we_parameter(vocab, '../data/glove_model.bin')
    else:
        opt.we_parameter = None

    # Load model module
    api = importlib.import_module(opt.model+'.api')

    # Construct the model
    model = api.get_model(opt)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            nDCG = checkpoint['nDCG']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, nDCG {})"
                  .format(opt.resume, start_epoch, nDCG))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            sys.exit()


    # Load valid loader
    val_loader = api.get_test_loader(vocab, opt, 'valid')

    if opt.test:
        # evaluate on validation set
        validate(api, opt, val_loader, model)
        sys.exit()

    # load train loader
    train_loader = api.get_train_loader(vocab, opt)

    # Train the Model
    best_nDCG = 0
    start = time.time()
    for epoch in range(opt.num_epochs):
        # train for one epoch
        train(opt, train_loader, model, epoch)

        # evaluate on validation set
        nDCG = validate(api, opt, val_loader, model, epoch)

        # update learning rate
        model.lr_step()

        # remember best R@ sum and save checkpoint
        is_best = nDCG > best_nDCG
        best_nDCG = max(nDCG, best_nDCG)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'nDCG': nDCG,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, path=opt.logger_name)
        print("best")
        print(best_nDCG)

    end = time.time()
    duration = int(end - start)
    minutes = (duration // 60) % 60
    hours = duration // 3600
    logging.info('Training time {}h {}min'.format(hours, minutes))


def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    print("epoch")
    print(epoch)
    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # Update the model
        global_step = epoch * len(train_loader) + i
        model.train_emb(global_step, *train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})'
                .format(
                    epoch+1, i+1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def validate( api, opt, val_loader, model, epoch=-1):
    # load answer
    answer = json.load(open(os.path.join('../data/kddcup', 'valid/valid_answer.json')))

    # scorer
    scorer = NDCGScorer(5)

    
    nDCG = api.evaluate(opt, model, val_loader, scorer, answer, log_step=opt.log_step, logging=logging.info)


    # print log
    print("nDCG score ", nDCG)
    logging.info('nDCG score {:.5f}'.format(nDCG))



    # record metrics in tensorboard, epoch=-1 when evaluation only
    if epoch != -1:
        tb_logger.log_value('nDCG', nDCG, epoch)

    return nDCG


#save checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', path=''):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
