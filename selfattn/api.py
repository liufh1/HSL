from . import data
from . import model
from . import evaluation

def get_train_loader(vocab, opt):

	return data.get_train_loader(vocab, opt)


def get_test_loader(vocab, opt, split, batch_size=4):

	return data.get_test_loader(vocab, opt, split, batch_size=batch_size)


def get_model(opt):

	return model.HSL(opt)


def evaluate(opt,model, val_loader, scorer, answer, k=5, log_step=50, logging=print):

	return evaluation.evaluate(model, val_loader, scorer, answer, log_step=log_step, logging=logging, out_path=opt.score_to)

