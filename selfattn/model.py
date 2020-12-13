import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from utils import *
from .loss import ContrastiveLoss
import torch.nn.functional as F
from mca import SA, AttFlat



class ImageEncoder(nn.Module):

    def __init__(self, opt, img_dim, embed_size, num_layers, d_model, no_imgnorm=False):
        super(ImageEncoder, self).__init__()
        self.no_imgnorm = no_imgnorm
        self.img_dim = img_dim
        #fc layers
        self.in_fc = nn.Linear(img_dim, d_model)
        self.out_fc = nn.Linear(d_model, embed_size)
        self.fc1 = nn.Linear(d_model, embed_size)
        self.fc2 = nn.Linear(embed_size, d_model)
        self.fc3 = nn.Linear(embed_size, d_model)
        self.dropout = nn.Dropout(0.1)
        self.attflat_img = AttFlat(opt)

        #transformer layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer1 = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.transformer2 = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

    def forward(self, x, num_boxes, x_mask):
        """Extract image feature vectors."""

        #input layer
        x = self.in_fc(x)

        mask = torch.zeros(x.size(0), x.size(1)) == 1
        for i in range(x.size(0)):
            mask[i,num_boxes[i]:] = True
        if torch.cuda.is_available():
            mask = mask.cuda()

        # 1st-level Forward propagate transformer
        x = x.transpose(0, 1)
        x = self.transformer1(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)

        img_feat = x

        # 2nd-level Forward propagate transformer
        x = x.transpose(0, 1)
        x = self.transformer2(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)

        #1st and 2nd level object level feature
        x2 = self.fc1(img_feat)
        x2 = self.dropout(x2)

        x1 = self.out_fc(x)
        x1 = self.dropout(x1)

        # 1st and 2nd level image level feature
        x3 = F.relu(x1)
        x3 = self.fc2(x3)
        x3 = self.attflat_img(x3, x_mask)
        
        x4 = F.relu(x2)
        x4 = self.fc3(x4)
        x4 = self.attflat_img(x4, x_mask)

        #x1 means 1st level object level feature
        #x2 means 2nd level object level feature
        #x3 means 1st level image level feature
        #x4 means 2nd level image level feature
        x1 = l2norm(x1)
        x2 = l2norm(x2)
        x3 = l2norm(x3)
        x4 = l2norm(x4)

        return x1,x2,x3,x4


class TextEncoder(nn.Module):

    def __init__(self, word_dim, embed_size, num_layers, no_txtnorm=False):
        super(TextEncoder, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=word_dim, nhead=4)
        self.transformer1 = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.transformer2 = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.fc1 = nn.Linear(word_dim, embed_size)
        self.fc2 = nn.Linear(word_dim, embed_size)
        self.fc3 = nn.Linear(word_dim, embed_size)
        self.fc4 = nn.Linear(word_dim, embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # generate key mask
        mask = torch.zeros(x.size(0), x.size(1)) == 1
        for i in range(x.size(0)):
            mask[i,lengths[i]:] = True
        if torch.cuda.is_available():
            mask = mask.cuda()

        x = x.transpose(0, 1)
        x = self.transformer1(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)

        out2 = []
        for i in range(x.size(0)):
            out2.append(torch.mean(x[i][:lengths[i]], dim=0))
        out2 = torch.stack(out2, dim=0)

        # Forward propagate transformer
        x = x.transpose(0, 1)
        x = self.transformer2(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)

        out = []
        for i in range(x.size(0)):
            out.append(torch.mean(x[i][:lengths[i]], dim=0))
        out = torch.stack(out, dim=0)

        out1 = self.fc1(out)
        out1 = self.dropout(out1)

        out3 = self.fc2(out2)
        out3 = self.dropout(out3)

        out4 = self.fc3(out)
        out4 = self.dropout(out4)

        out5 = self.fc4(out2)
        out5 = self.dropout(out5)
    

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            out1 = l2norm(out1)
            out3 = l2norm(out3)
            out4 = l2norm(out4)
            out5 = l2norm(out5)

        return out1, out3, out4, out5

class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, word_dim, we_parameter):
        super(WordEmbedding, self).__init__()
        self.we_parameter = we_parameter is not None
        if we_parameter is not None:
            self.embed = nn.Embedding(vocab_size, 300)
            self.fc = nn.Linear(300, word_dim)
        else:
            self.embed = nn.Embedding(vocab_size, word_dim)

        self.init_weights(we_parameter)

    def forward(self, x):
        x = self.embed(x)
        if self.we_parameter:
            x = self.fc(x)
        return x

    def init_weights(self, we_parameter):
        if we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)


class HSL(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip
        self.score_agg = opt.score_agg
        self.img_enc = ImageEncoder(opt, opt.img_dim, opt.embed_size,
                                    opt.img_num_layers, opt.d_model, opt.no_imgnorm)
        self.txt_enc = TextEncoder(opt.word_dim, opt.embed_size, 
                                    opt.txt_num_layers, opt.no_txtnorm)
        self.word_embed = WordEmbedding(opt.vocab_size, opt.word_dim, opt.we_parameter)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.word_embed.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss()
        self.criterion2 = ContrastiveLoss()
        self.criterion3 = ContrastiveLoss()
        self.criterion4 = ContrastiveLoss()

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.word_embed.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.step_size, gamma=opt.decay_rate)

        self.Eiters = 0

        self.logger = LogCollector()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.word_embed.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.word_embed.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode
        """
        self.Eiters = 0
        self.img_enc.train()
        self.txt_enc.train()
        self.word_embed.train()
        self.logger.reset()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.word_embed.eval()

    def forward_emb(self, images, num_boxes, captions, lengths, no_grad=False):
        """Compute the image and caption embeddings
            img_emb: batch_size, max_boxes, e_dim
            cap_emb: batch_size, e_dim
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images, num_boxes, captions, lengths = images.cuda(), num_boxes.cuda(), captions.cuda(), lengths.cuda()

        image_mask = self.make_mask(images)
        # Forward
        if no_grad:
            with torch.no_grad():
                captions = self.word_embed(captions)
                img_emb, img_emb2, img_emb3, img_emb4 = self.img_enc(images, num_boxes, image_mask)
                cap_emb, cap_emb2, cap_emb3, cap_emb4 = self.txt_enc(captions, lengths)
        else:
            captions = self.word_embed(captions)
            img_emb, img_emb2, img_emb3, img_emb4 = self.img_enc(images, num_boxes, image_mask)
            cap_emb, cap_emb2, cap_emb3, cap_emb4 = self.txt_enc(captions, lengths)

        return img_emb, cap_emb, img_emb2, cap_emb2, img_emb3, cap_emb3, img_emb4, cap_emb4

    def forward_loss(self, sims, sims2, sims3, sims4, global_step, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        # compute image-sentence score matrix

        loss = self.criterion(sims)
        loss2 = self.criterion2(sims2)
        loss3 = self.criterion3(sims3)
        loss4 = self.criterion4(sims4)
        loss_total = loss + 0.5 * loss2 + loss3 + 0.5 * loss4
        #return loss
        return loss_total
        #return loss

    def train_emb(self,  global_step, images, num_boxes, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, img_emb2, cap_emb2, img_emb3, cap_emb3, img_emb4, cap_emb4 = self.forward_emb(images, num_boxes, captions, lengths)

        scores = self.cal_scores(img_emb, cap_emb, num_boxes)
        scores2 = self.cal_scores(img_emb2, cap_emb2, num_boxes)
        scores3 = self.cal_scores2(img_emb3, cap_emb3, num_boxes)
        scores4 = self.cal_scores2(img_emb4, cap_emb4, num_boxes)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(scores,scores2,scores3,scores4, global_step)

        self.logger.update('Cnt Le', loss.cpu().item(), scores.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

    def lr_step(self):
        self.lr_scheduler.step()

    def cal_scores(self, img_emb, cap_emb, num_boxes):
        # => batch_size*max_boxes, embed_size
        img_emb = img_emb.contiguous()
        img_emb = img_emb.view(img_emb.size(0)*img_emb.size(1), -1)
        # => batch_size, batch_size*max_boxes
        sims = torch.mm(cap_emb, img_emb.t())
        # => batch_size(cap), batch_size(img), max_boxes
        sims = sims.view(sims.size(0), sims.size(0), -1)
        # => batch_size(img), batch_size(cap), max_boxes
        sims = sims.transpose(0, 1)

        scores = []
        # for each image
        for i in range(sims.size(0)):
            if self.score_agg == 'max':
                scores.append(sims[i][:,:num_boxes[i]].max(dim=1)[0])
            elif self.score_agg == 'mean':
                scores.append(sims[i][:,:num_boxes[i]].mean(dim=1))
            elif self.score_agg == 'sum':
                scores.append(sims[i][:,:num_boxes[i]].sum(dim=1))
            elif self.score_agg.startswith('top'):
                k = int(self.score_agg[-1])
                scores.append(torch.sort(sims[i][:,:num_boxes[i]], 1, descending=True)[0][:,:k].mean(dim=1))
        scores = torch.stack(scores, dim=1)

        return scores

    #calculate aggregated feature's similarity
    def cal_scores2(self, img_emb, cap_emb, num_boxes):
        scores = torch.mm(cap_emb, img_emb.t())
        return scores

    def make_mask(self, feature):
        mask = (torch.sum(torch.abs(feature), dim=-1) == 0)
        mask = mask.unsqueeze(1).unsqueeze(2)
        return mask