import torch.nn.functional as F
from transformer import *


class Translator(nn.Module):
    def __init__(self, model, beam_size, max_seq_len, trg_bos_idx, trg_eos_idx, device):
        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx
        self.src_pad_idx = 0
        self.trg_pad_idx = 0
        self.model = model
        self.register_buffer('init_seq', torch.tensor([[trg_bos_idx]], dtype=torch.long, device=device))
        self.register_buffer('blank_seqs', torch.full((beam_size, max_seq_len),
                                                      self.trg_pad_idx,
                                                      dtype=torch.long,
                                                      device=device))
        self.register_buffer('len_map', torch.arange(1, max_seq_len+1, dtype=torch.long, device=device).unsqueeze(0))
        self.blank_seqs[:, 0] = trg_bos_idx
        self.device = device

    def _model_decoder(self, trg_seq, enc_output, src_mask, trg_mask):
        dec_output = self.model.decoder(trg_seq, enc_output, src_mask, trg_mask)
        dec_output = self.model.generator(dec_output)
        return F.softmax(dec_output, dim=-1)

    def _get_init_state(self, src_seqs, src_mask, trg_mask):
        beam_size = self.beam_size
        batch_size = src_seqs.size(0)
        src_seqs = src_seqs.to(self.device)
        src_mask = src_mask.to(self.device)
        trg_mask = trg_mask.to(self.device)
        enc_outputs = self.model.encoder(src_seqs, src_mask)

        dec_outputs = self._model_decoder(self.init_seq.expand(batch_size, -1), enc_outputs, src_mask, trg_mask)
        best_k_probs, best_k_idx = dec_outputs[:, -1, :].topk(beam_size)
        scores = torch.log(best_k_probs).view(batch_size, beam_size)
        # self.blank_seqs = self.blank_seqs.unsqueeze(0).repeat(batch_size, 1, 1).view(beam_size*batch_size, -1)
        gen_seqs = self.blank_seqs.unsqueeze(0).repeat(batch_size, 1, 1).view(beam_size*batch_size, -1).clone().detach()
        gen_seqs[:, 1] = best_k_idx.squeeze(1).view(batch_size*beam_size, -1).squeeze(-1)

        enc_outputs = enc_outputs.repeat_interleave(beam_size, dim=0)
        return enc_outputs, gen_seqs, scores

    def _get_the_best_score_and_idx(self, gen_seqs, dec_outputs, scores, step):
        assert len(scores.size()) == 2
        beam_size = self.beam_size
        batch_size = gen_seqs.size(0) // beam_size
        vocab_size = dec_outputs.size(-1)
        dec_outputs = dec_outputs.view(batch_size, beam_size, -1, vocab_size)
        best_k2_probs, best_k2_idx = dec_outputs[:, :, -1, :].topk(beam_size)
        scores = torch.log(best_k2_probs) + scores.view(batch_size, beam_size, 1)
        scores, best_k_idx_in_k2 = scores.view(batch_size, -1).topk(beam_size)

        best_k_r_idx, best_k_c_idx = torch.div(best_k_idx_in_k2, beam_size).long(), \
                                     torch.remainder(best_k_idx_in_k2, beam_size).long()

        best_k_idx = best_k2_idx[torch.arange(batch_size).unsqueeze(1), best_k_r_idx, best_k_c_idx]

        gen_seqs = gen_seqs.view(batch_size, beam_size, -1)
        gen_seqs[:, :, :step] = torch.gather(gen_seqs[:, :, :step], 1, best_k_r_idx.unsqueeze(-1).expand(-1, -1, step))
        gen_seqs[:, :, step] = best_k_idx.view(batch_size, -1)
        return gen_seqs.view(batch_size * beam_size, -1), scores

    def translate(self, src_seqs):
        with torch.no_grad():
            batch_size = src_seqs.size(0)
            beam_size = self.beam_size
            src_mask = Transformer.make_pad_mask(src_seqs, self.src_pad_idx, device=self.device)
            trg_mask = Transformer.make_no_peak_mask(self.init_seq, device=self.device)
            enc_outputs, gen_seqs, scores = self._get_init_state(src_seqs, src_mask, trg_mask)
            src_mask = src_mask.repeat_interleave(beam_size, dim=0)
            eos_flag = False
            for step in range(2, self.max_seq_len):
                trg_mask = Transformer.make_no_peak_mask(gen_seqs[:, :step], device=self.device)
                dec_outputs = self._model_decoder(gen_seqs[:, :step], enc_outputs, src_mask, trg_mask)
                gen_seqs, scores = self._get_the_best_score_and_idx(gen_seqs, dec_outputs, scores, step)
                eos_locs = gen_seqs == self.trg_eos_idx
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, self.max_seq_len).min(1)
                _, ans_idx = scores.div(seq_lens.view(batch_size, -1).float() ** self.alpha).max(1)
                if (eos_locs.sum(1) > 0).sum(0).item() == self.beam_size*batch_size:
                    _, ans_idx = scores.div(seq_lens.view(batch_size, -1).float() ** self.alpha).max(1)
                    eos_flag = True
                    break
            if not eos_flag:
                _, ans_idx = scores.div(seq_lens.view(batch_size, -1).float() ** self.alpha).max(1)

            gen_seqs_batch = gen_seqs.view(batch_size, beam_size, -1)
            seq_lens_batch = seq_lens.view(batch_size, -1)
            pred_all = torch.gather(gen_seqs_batch, 1,
                                    ans_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,
                                                                               gen_seqs_batch.size(2))).squeeze(1)
            len_idx = torch.gather(seq_lens_batch, 1, ans_idx.unsqueeze(-1).expand(-1, -1)).squeeze(-1)
            res = []
            for i in range(len(len_idx)):
                res.append(pred_all[i, :len_idx[i].item()].tolist())

            return res



