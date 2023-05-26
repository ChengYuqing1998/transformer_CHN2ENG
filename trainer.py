import pickle
import tqdm
import wandb
import os
from torch.optim import lr_scheduler
import logging
from translator import *
from tqdm import tqdm
from torch.nn.functional import log_softmax
import nltk


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def build_trainer(model, configs):
    trainer = Trainer(model=model,
                      ckpt_dir=configs['ckpt_dir'],
                      ckpt_file_name=configs['ckpt_file_name'],
                      log_dir=configs['log_dir'],
                      log_file_name=configs['log_file_name'],
                      print_freq=configs['print_freq'],
                      eval_freq=configs['eval_freq'],
                      save_freq=configs['save_freq'],
                      device=eval(configs['device']),
                      write_config=configs
                      )
    trainer.build_loss(configs['loss_type'], configs['smoothing'], configs['ignore_index'])
    trainer.build_optimizer(configs['learning_rate'], configs['optimizer_type'])
    if configs['scheduler_flag']:
        trainer.build_scheduler(configs['anneal_rate'],
                                configs['scheduler_type'],
                                configs['patience'],
                                configs['threshold'])
    return trainer


def remove_element(lst, element):
    if isinstance(lst, list):
        return [remove_element(sublst, element) for sublst in lst if sublst != element]
    else:
        return lst if lst != element else None


class Trainer:
    def __init__(self, model: nn.Module, ckpt_dir, ckpt_file_name, log_dir, log_file_name,
                 print_freq, eval_freq, save_freq, device, write_config, **kwargs):
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.ckpt_file_name = ckpt_file_name
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        self.print_freq = print_freq
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.device = device
        self.write_config = write_config

        self.optimizer = None
        self.loss = None
        self.scheduler = None
        self.global_step = 0
        self.best_loss = 1e9

        self.log = self.make_log(log_dir, log_file_name)
        self.log.info(msg=self.write_config)

    def build_optimizer(self, learning_rate, optimizer_type):
        assert optimizer_type in ['sgd', 'adam']
        if optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def build_scheduler(self, anneal_rate, scheduler_type, patience, threshold):
        assert scheduler_type in ['exp', 'plateau', 'cosine']
        if scheduler_type == 'exp':
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, anneal_rate)
        elif scheduler_type == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            mode='min',
                                                            patience=patience,
                                                            factor=anneal_rate,
                                                            threshold=threshold,
                                                            threshold_mode='abs')
        elif scheduler_type == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                            T_max=self.write_config['max_epochs'],
                                                            eta_min=self.write_config['learning_rate'] * 1e-2)

    def build_loss(self, loss_type, smoothing, ignore_index=None, **kwargs):
        assert loss_type in ['ce', 'nll', 'kl']
        if loss_type == 'ce':
            self.loss= torch.nn.CrossEntropyLoss(ignore_index=int(ignore_index))
            # self.loss = LabelSmoothing(self.model.dec_voc_size, int(ignore_index), criterion, smoothing)
        elif loss_type == 'nll':
            self.loss = torch.nn.NLLLoss(ignore_index=int(ignore_index))
        elif loss_type == 'kl':
            self.loss = LabelSmoothing(self.model.dec_voc_size, int(ignore_index), smoothing)

    def fit(self, train_data, val_data, input_lang, output_lang, max_epochs, warmup, clip=None, dict_flag=False, **kwargs):
        self.model.train()
        for epoch in tqdm(range(1, max_epochs+1)):
            train_loss_in_epoch = []
            for x, y in train_data:
                loss = self.fit_iter(x, y, clip=clip)
                train_loss_in_epoch.append(loss)
                self.log.info(msg=('Epoch:', '%04d' % epoch, 'Training Loss =', '{:.6f}'.format(loss)))
                wandb.log({'epoch': epoch, 'train_loss': loss})
                if self.global_step % self.print_freq == 0:
                    avg_loss = sum(train_loss_in_epoch) / len(train_loss_in_epoch)
                    print('Epoch:', '%04d' % epoch, 'Average Training Loss =', '{:.6f}'.format(avg_loss))

            if epoch == 1:
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print("total_para_counts: ", total_params)
                wandb.log({'total_para_counts': total_params})

            if epoch % self.save_freq == 0:
                self._save(self.model, self.ckpt_dir, self.ckpt_file_name, dict_flag=dict_flag)

            if epoch % self.eval_freq == 0:
                eval_loss, eval_bleu4 = self.eval(val_data, input_lang, output_lang)
                self.log.info(msg=('Epoch:', '%04d' % epoch, 'Evaluating Loss =', '{:.6f}'.format(eval_loss),
                                   'Evaluating Bleu4 =', '{:.6f}'.format(eval_bleu4)))
                wandb.log({'epoch': epoch, 'val_loss': eval_loss, 'val_bleu4': eval_bleu4})
                print('Epoch:', '%04d' % epoch, 'Evaluating Loss =', '{:.6f}'.format(eval_loss),
                      'Evaluating Bleu4 =', '{:.6f}'.format(eval_bleu4))
                if eval_loss < self.best_loss:
                    last_file_name = 'intermediate_model' + '-' + self.ckpt_file_name + '-{:.4f}.pt'.format(
                        self.best_loss)
                    last_file_path = os.path.join(self.ckpt_dir + '/intermediate', last_file_name)
                    if os.path.exists(last_file_path):
                        os.remove(last_file_path)
                    self.best_loss = eval_loss
                    self._save(self.model, self.ckpt_dir + '/intermediate',
                               'intermediate_model' + '-' + self.ckpt_file_name + '-{:.4f}.pt'.format(eval_loss),
                               dict_flag=True)

            if self.scheduler and epoch > warmup and self.eval_freq == 1:
                self.log.info(msg=('=====starting plateau====='))
                self.scheduler.step(eval_loss)

    def fit_iter(self, x, y, clip=None):
        self.global_step += 1
        self.optimizer.zero_grad()
        x = x.to(self.device)
        y = y.to(self.device)
        # y_pred = self.model.forward(x, y[:, :-1])
        y_pred = log_softmax(self.model.forward(x, y[:, :-1]), dim=-1)

        loss = self.loss(y_pred.contiguous().view(-1, y_pred.shape[-1]), y[:, 1:].contiguous().view(-1))
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]['lr']
        wandb.log({'global_step': self.global_step})
        wandb.log({'learning_rate': lr})
        return loss

    def eval(self, val_data, input_lang, output_lang):
        self.model.eval()
        translator = Translator(self.model, 3, self.model.max_len, self.model.trg_bos_idx,
                                self.model.trg_eos_idx, device=self.device)
        validate_loss = 0.0
        bleu4 = 0.0
        sample_num = 0
        with torch.no_grad():
            for x, y in tqdm(val_data):
                x = x.to(self.device)
                y = y.to(self.device)
                sample_num += len(x)
                y_pred = log_softmax(self.model.forward(x, y[:, :-1]), dim=-1)
                loss = self.loss(y_pred.contiguous().view(-1, y_pred.shape[-1]), y[:, 1:].contiguous().view(-1))
                res = translator.translate(x)
                refer = y.unsqueeze(1).cpu().numpy().tolist()
                refer_clean = remove_element(refer, 0)
                bleu4_batch = nltk.translate.bleu_score.corpus_bleu(refer_clean, res, weights=(0.25, 0.25, 0.25, 0.25))
                bleu4 += bleu4_batch * len(x)
                validate_loss += loss.item() * len(x)
                # break

            enc_input = torch.tensor([[1, 15, 198, 36, 280, 485, 65, 258, 37, 13, 80, 21, 2, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            pred_seqs = translator.translate(enc_input)
            input_sentence = ' '.join(remove_element([input_lang.index2word[_] for _ in enc_input.squeeze(0).cpu().numpy()], '<pad>'))
            print(input_sentence)
            output_sentence = [' '.join([output_lang.index2word[_] for _ in pred_seq]) for pred_seq in pred_seqs][0]
            print("pred seq: ", output_sentence)
            print("true label: do you really believe that s what happened ?")

        return validate_loss / sample_num, bleu4 / sample_num

    def _save(self, model, ckpt_dir, ckpt_file_name, dict_flag=False):
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        save_path = os.path.join(ckpt_dir, ckpt_file_name)
        if not dict_flag:
            torch.save(model, save_path)
        else:
            torch.save(model.state_dict(), save_path)

    @staticmethod
    def make_log(log_dir, log_file_name):
        path = os.path.join(log_dir, log_file_name)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if os.path.exists(path):
            mode = 'a'
        else:
            mode = 'w'
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            filename=path,
                            filemode=mode)
        return logging
