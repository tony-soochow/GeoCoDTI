

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gc
import datetime
import sys
import subprocess
import pickle
from xml.parsers.expat import model
from torch.utils.data import DataLoader
from utils_mine import *
from esm_model import *
from few_dataloader import *
import itertools
import random
import time
import pdb
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from torch.serialization import safe_globals
warnings.filterwarnings("ignore")


def main(args, start_time):
    def auto_select_device(use_gpu, gpu_id, remap, required_mem_min=2000, strategy='random'):

        def get_gpu_memory_map(remap):
            result = subprocess.check_output(
                [
                    'nvidia-smi', '--query-gpu=memory.used',
                    '--format=csv,nounits,noheader'
                ], encoding='utf-8')
            gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
            return gpu_memory[remap]

        def get_total_gpu_memory_map(remap):
            result = subprocess.check_output(
                [
                    'nvidia-smi', '--query-gpu=memory.total',
                    '--format=csv,nounits,noheader'
                ], encoding='utf-8')
            gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
            return gpu_memory[remap]

        if torch.cuda.is_available() and use_gpu:
            if gpu_id == -1:
                total_memory_raw = get_total_gpu_memory_map(remap)
                memory_raw = get_gpu_memory_map(remap)
                available_memory = total_memory_raw - memory_raw
                available_memory[0] = available_memory[0] - 24000
                unavailable_gpu = []
                for i, m in enumerate(available_memory):
                    if m < required_mem_min:
                        unavailable_gpu.append(i)
                print('Total GPU Mem: {}'.format(total_memory_raw))
                print('Available GPU Mem: {}'.format(available_memory))
                print('Unselectable GPU ID: {}'.format(unavailable_gpu))
                if strategy == 'random':
                    memory = available_memory / available_memory.sum()
                    memory[unavailable_gpu] = 0
                    gpu_prob = memory / memory.sum()
                    cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
                    print('GPU Prob: {}'.format(gpu_prob.round(2)))
                    print(
                        'Random select GPU, select GPU {} with mem: {}'.format(
                            cuda, available_memory[cuda]))
                elif strategy == "max":
                    available_memory[unavailable_gpu] = 0
                    cuda = np.argmax(available_memory)
                    print(
                        'Max select GPU, select GPU {} with mem: {}'.format(
                            cuda, available_memory[cuda]))
                else:
                    raise AssertionError
                return torch.device('cuda:{}'.format(cuda))
            elif 0 <= gpu_id <= 9:
                print(
                    'Manually select GPU, select GPU {}'.format(
                        gpu_id))
                return torch.device('cuda:{}'.format(gpu_id))
            else:
                raise ValueError("The GPU id is invalid！")
        else:
            print('cuda not available')
            return torch.device("cpu")

    class EarlyStopping:

        def __init__(self, directory, save, save_thre, save_mul, fun, patience=15,
                     ):
            self.patience = patience
            self.counter = 0
            self.best_score = float('-inf')
            self.early_stop = False
            self.directory = directory
            self.best_model_path = None
            self.trace_fun = print
            self.save = save
            self.save_thre = save_thre
            self.save_mul = save_mul
            self.fun = fun

        def __call__(self, score, model):

            if (1 - score) >= (1 - self.best_score) * 0.97:
                if_opt = False
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.trace_fun("Training early stopped!")
            else:
                if_opt = True
                self.best_score = score
                self.counter = 0
                self.trace_fun("*****************new opt*******************")
                if self.save and score>self.save_thre:
                    if self.save_mul:
                        save2file(model, self.directory, start_time, info=self.fun + "_" + str(score))
                    else:
                        save2file(model, self.directory, start_time, info=self.fun)
                    self.trace_fun("model saved")


            return self.early_stop, if_opt

    def train_loop(kge_model, optimizer, warm_up_steps, train_iterator, train_step_fun, if_valid=False,
                   valid_dataloader=None, valid_labels=None, test_loop=None, test_module_name=None, test_dataloader=None, test_labels=None):
        current_learning_rate = args.learning_rate
        opt_aupr = float("-inf")
        early_stop_count = 0
        model_path = str()
        try:
            for step in range(args.max_relation_steps):
                loss = train_step_fun(kge_model, optimizer, train_iterator, args)
                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, kge_model.parameters()),
                        lr=current_learning_rate,
                        weight_decay=args.weight_decay
                    )
                    warm_up_steps = warm_up_steps * 3

                if if_valid:
                    
                    if (step + 1) % (args.relation_valid_steps // 1) == 0:
                        if test_module_name == "relation":
                            _, acc, auc, aupr = test_loop(args, kge_model, valid_dataloader,
                                                          valid_labels)
                        else:
                            raise ValueError


                        print(f"ACC: {acc:.5f}  AUC: {auc:.5f}  AUPR: {aupr:.5f}")

                        if (1 - aupr) < (1 - opt_aupr) * 0.97:
                            opt_aupr = aupr
                            early_stop_count = 0
                            print("***************new opt***************")
                            if 0 < aupr and args.save:
                                save2file(kge_model, f"{args.model_path}/", start_time, info="kge")
                                print("Model saved!")
                            _, test_acc, test_auc, test_aupr = test_loop(args, kge_model, test_dataloader,
                                                                         test_labels)
                            print(f"test ACC: {test_acc:.5f} test AUC: {test_auc:.5f}  AUPR: {test_aupr:.5f}#")

                        else:
                            early_stop_count += 1
                            if early_stop_count >= 8:
                                print("Early stopped!")
                                return kge_model, test_acc, test_auc, test_aupr
        except KeyboardInterrupt:
            pass
    def load_model_to_device(model_path, device, load_kge_model=None, load_self_model=None, load_rel_model=None, esm_model=None):
        print("load model from disk...")
        device = int(device)
        if load_kge_model:
            print(f"load kge model from {model_path}/../{load_kge_model}")
            kwargs, state = torch.load(f"{model_path}/../{load_kge_model}", map_location=torch.device(device))
            model = KGEModel(**kwargs)
        elif load_self_model:
            kwargs, state = torch.load(f"{model_path}/{load_self_model}", map_location=torch.device(device))
            model = SelfModel(kwargs["hidden_dim"], esm_model)
        elif load_rel_model:
            kwargs, state = torch.load(f"{model_path}/{load_rel_model}", map_location=torch.device(device))
            model = RelationModel(entity_dim=kwargs["entity_dim"], entity_embeddings=state["entity_embeddings"])
        else:
            raise ValueError
        
        model.load_state_dict(state, strict=False)
        return model.to(device)

    def train_kge(args):


        warm_up_steps = args.warm_up_steps

        if args.baseline:
            print("Using baseline model")
            kg_triples = read_triple(os.path.join(f"{args.data_path}/../", 'drkg.tsv'), entity2id,
                                        relation2id)
            train_triples = read_triple(os.path.join(args.data_path, 'train.tsv'), entity2id,
                                        relation2id)
            aug_times = len(kg_triples) // len(train_triples)
            train_triples = train_triples * aug_times + kg_triples
        else:
            train_triples = read_triple(os.path.join(f"{args.data_path}/../", 'drkg.tsv'), entity2id,
                                        relation2id)
        valid_triples = read_triple(os.path.join(args.data_path, 'valid.tsv'), entity2id, relation2id)
        test_triples = read_triple(os.path.join(args.data_path, 'test.tsv'), entity2id, relation2id)
        test_neg_triples = read_triple(os.path.join(args.data_path, 'test_neg.tsv'), entity2id, relation2id)
        valid_neg_triples = read_triple(os.path.join(args.data_path, 'valid_neg.tsv'), entity2id, relation2id)

        valid_labels = torch.cat((torch.ones(len(valid_triples)), torch.zeros(len(valid_neg_triples))))
        test_labels = torch.cat(
            (torch.ones(len(test_triples)), torch.zeros(len(test_neg_triples))))

        train_relation_dataloader_head = DataLoader(
            RelationPretrainDataset(train_triples, n_entity, n_relation, args.negative_sample_size, 'head-batch'),
            batch_size=args.relation_batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 20),
            collate_fn=RelationPretrainDataset.collate_fn,
        )
        train_relation_dataloader_tail = DataLoader(
            RelationPretrainDataset(train_triples, n_entity, n_relation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.relation_batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 20),
            collate_fn=RelationPretrainDataset.collate_fn
        )
        train_relation_iterator = BidirectionalOneShotIterator(train_relation_dataloader_head,
                                                               train_relation_dataloader_tail)
        test_relation_dataloader = DataLoader(
            TestRelationPretrainDataset(
                test_triples + test_neg_triples,
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 20),
            collate_fn=TestRelationPretrainDataset.collate_fn,
            shuffle=False
        )
        valid_relation_dataloader = DataLoader(
            TestRelationPretrainDataset(
                valid_triples + valid_neg_triples,
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 20),
            collate_fn=TestRelationPretrainDataset.collate_fn,
            shuffle=False
        )

        kge_model = KGEModel(
            model_name=args.model,
            nentity=n_entity,
            nrelation=n_relation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            double_entity_embedding=args.double_entity_embedding,
            double_relation_embedding=args.double_relation_embedding
        )
        kge_model = kge_model.to(args.device)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        kge_model, test_acc, test_auc, test_aupr = train_loop(kge_model, optimizer, warm_up_steps, train_relation_iterator, kge_model.train_relation_step,
                   True,
                   valid_relation_dataloader, valid_labels, kge_model.evaluate, "relation", test_relation_dataloader, test_labels)

        return kge_model, test_acc, test_auc, test_aupr

    def gen_psd_label(fun, args, model=None, esm_converter=None, ent_id2seq=None, ent_id2smiles=None):
            
        def gen_cartesian_triples(relation2id, ent_id2seq, ent_id2smiles, cartesian_ratio):
            cmp_with_smiles = tuple(ent_id2smiles.keys())
            gene_with_seq = tuple(ent_id2seq.keys())
            rel_id = relation2id["DTI"]

            n_cmp = len(cmp_with_smiles)
            n_gene = len(gene_with_seq)
            total = n_cmp * n_gene

            if total == 0:
                return list()

            sample_num = int(total * cartesian_ratio)
            sample_num = min(sample_num, total)
            sampled_flat_indices = random.sample(range(total), sample_num)

            cmp_gene_cartesian_product = [
                (cmp_with_smiles[idx // n_gene], rel_id, gene_with_seq[idx % n_gene])
                for idx in sampled_flat_indices
            ]

            print(len(cmp_gene_cartesian_product))
            print(cmp_gene_cartesian_product[0:5])
            return cmp_gene_cartesian_product

        
        cartesian_ratio = args.kge_cartesian_ratio if fun == "psd_label_kge" else args.self_cartesian_ratio
        cartesian_triples = gen_cartesian_triples(relation2id, ent_id2seq, ent_id2smiles, cartesian_ratio)
        
        pseudo_num_workers = max(2, 8)
        test_cartesian_dataloader = DataLoader(
            SelfDataset(
                cartesian_triples, list(), ent_id2smiles, ent_id2seq
            ),
            batch_size= 32,
            shuffle=False,
            num_workers=pseudo_num_workers,
            pin_memory=(torch.cuda.is_available() and int(args.device) >= 0),
            persistent_workers=True,
            collate_fn=SelfDataset.collate_fn,
        )
        support_pos_triples = read_triple(os.path.join(args.data_path, 'train.tsv'), entity2id, relation2id)
        support_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id, relation2id)
        support_dataloader = DataLoader(
            SelfDataset(
                support_pos_triples, support_neg_triples, ent_id2smiles, ent_id2seq
            ),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=pseudo_num_workers,
            pin_memory=(torch.cuda.is_available() and int(args.device) >= 0),
            persistent_workers=True,
            collate_fn=SelfDataset.collate_fn,
        )

        def _forward_feature_vec(smiles_feats, seq_feats, heads, tails, device):
            if isinstance(model, RelationModel):
                heads_dev = heads.to(device)
                tails_dev = tails.to(device)
                return model(heads_dev, tails_dev)
            elif isinstance(model, SelfModel):
                smiles_feats_dev = (
                    smiles_feats[0].to(device),
                    smiles_feats[1].to(device)
                )
                seq_feats_dev = seq_feats.to(device)
                return model(smiles_feats_dev, seq_feats_dev)
            else:
                raise ValueError("Unsupported model type for pseudo-label generation")

        def _build_prototypes(loader, device):
            model.eval()
            pos_feats, neg_feats = list(), list()
            with torch.no_grad():
                for smiles_feats, seq_feats, labels, heads, tails in loader:
                    logits = _forward_feature_vec(smiles_feats, seq_feats, heads, tails, device).detach().cpu()
                    labels = labels.cpu()
                    if (labels == 1).any():
                        pos_feats.append(logits[labels == 1])
                    if (labels == 0).any():
                        neg_feats.append(logits[labels == 0])

            if len(pos_feats) == 0 or len(neg_feats) == 0:
                raise ValueError("Prototype construction failed: support set lacks positive or negative samples")

            pos_proto = torch.cat(pos_feats, dim=0).mean(dim=0, keepdim=True)
            neg_proto = torch.cat(neg_feats, dim=0).mean(dim=0, keepdim=True)
            return pos_proto, neg_proto

        def _predict_by_prototype(eval_loader, device, pos_proto, neg_proto, temperature=1.0):
            model.eval()
            selected_triples = list()
            total_batches = len(eval_loader)
            log_every = max(1, total_batches // 10)
            eps = 1e-8
            temperature = max(float(temperature), eps)

            pos_proto = pos_proto.to(device)
            neg_proto = neg_proto.to(device)

            with torch.no_grad():
                for batch_idx, (smiles_feats, seq_feats, _, heads, tails) in enumerate(eval_loader):
                    logits = _forward_feature_vec(smiles_feats, seq_feats, heads, tails, device)

                    probs_pos = nnfun.softmax(logits, dim=1)[:, 1]

                    d_pos = torch.norm(logits - pos_proto, p=2, dim=1)
                    d_neg = torch.norm(logits - neg_proto, p=2, dim=1)

                    d_pos = d_pos / temperature
                    d_neg = d_neg / temperature
                    uncertainty = 1.0 - (torch.abs(d_neg - d_pos) / (d_neg + d_pos + eps))

                    probs_pos_cpu = probs_pos.cpu()
                    uncertainty_cpu = uncertainty.cpu()
                    batch_start = batch_idx * 32
                    batch_end = min(batch_start + probs_pos_cpu.shape[0], len(cartesian_triples))
                    for local_i, global_i in enumerate(range(batch_start, batch_end)):
                        prob_mean = probs_pos_cpu[local_i].item()
                        unc = uncertainty_cpu[local_i].item()
                        if unc <= args.pseudo_uncertainty_max and (
                                prob_mean >= args.pseudo_pos_threshold or prob_mean <= args.pseudo_neg_threshold):
                            selected_triples.append((cartesian_triples[global_i], prob_mean, unc))

                    if ((batch_idx + 1) % log_every == 0) or (batch_idx == total_batches - 1):
                        print(f"[{fun}] Prototype chunk {batch_idx + 1}/{total_batches} finished")

            return selected_triples

        print("Start prototype-distance pseudo labeling")
        pos_proto, neg_proto = _build_prototypes(support_dataloader, args.device)
        triple_preds = _predict_by_prototype(
            test_cartesian_dataloader,
            args.device,
            pos_proto,
            neg_proto,
            temperature=args.prototype_temperature
        )
        print("End prototype-distance evaluate~~~~~~~~~~~")

        triple_preds.sort(key=lambda x: x[1], reverse=True)
        print(f"Generated {len(triple_preds)} triples")
        if args.save:
            save2file(triple_preds, args.pkl_path, start_time, info=fun)
        return triple_preds
    
    def train_module(fun, args, triple_preds=None, kge_model=None, self_model=None, rel_model=None, max_ephs=999999):
        def read_psd_triples(true_triple_len, psd_use_ratio, balance_psd, triple_preds, balance_ratio):
            pos_candidates, neg_candidates = list(), list()
            for tri_pred in triple_preds:
                if len(tri_pred) == 2:
                    tri, prob = tri_pred
                    unc = 0.0
                else:
                    tri, prob, unc = tri_pred

                if unc > args.pseudo_uncertainty_max:
                    continue
                if prob >= args.pseudo_pos_threshold:
                    pos_candidates.append((tri, prob, unc))
                elif prob <= args.pseudo_neg_threshold:
                    neg_candidates.append((tri, prob, unc))

            pos_candidates.sort(key=lambda x: x[1], reverse=True)
            neg_candidates.sort(key=lambda x: x[1])

            print(f"pos candidates: {len(pos_candidates)}, neg candidates: {len(neg_candidates)}")

            psd_use_num = args.pseudo_topk_per_class
            psd_use_num = max(1, psd_use_num)
            if args.pseudo_topk_per_class > 0:
                psd_use_num = min(psd_use_num, args.pseudo_topk_per_class)
            psd_use_num = min(psd_use_num, len(pos_candidates))
            
            psd_triples = [pos_candidates[i] for i in range(psd_use_num)]
            if balance_psd:
                balance_times = psd_use_num / true_triple_len if true_triple_len > 0 else 1
                print(f"Balance_times: {balance_times}")
                if balance_times < 1:
                    balance_times = 1
                    print("psd triples are fewer than real triples")
                else:
                    balance_times = int(balance_times*balance_ratio)
                    if balance_times < 1:
                        balance_times = 1
            else:
                balance_times = 1

            neg_need_num = psd_use_num + (balance_times - 1) * true_triple_len
            if args.pseudo_topk_per_class > 0:
                neg_need_num = min(neg_need_num, args.pseudo_topk_per_class)
            neg_need_num = min(neg_need_num, len(neg_candidates))
            psd_neg_triples = [neg_candidates[i] for i in range(neg_need_num)]

            print(f"psd_triples num: {len(psd_triples)}, psd_neg_triples num: {len(psd_neg_triples)}")
            
            print(f"balance_times: {balance_times}")
            return psd_triples, psd_neg_triples, balance_times

        class WeightedSelfDataset(Dataset):
            def __init__(self, train_pos_meta, train_neg_meta, ent_id2smiles, ent_id2seq):
                self.samples = list(train_pos_meta) + list(train_neg_meta)

                self.ent_id2smiles = ent_id2smiles
                self.ent_id2seq = ent_id2seq

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                tri, prob, unc = self.samples[idx]
                head, _, tail = tri
                smiles_feat = self.ent_id2smiles[head]
                seq_feat = self.ent_id2seq[tail]
                return smiles_feat, seq_feat, head, tail, prob, unc

            @staticmethod
            def collate_fn(batch):
                smiles_gs = dgl.batch([_[0][0] for _ in batch])
                smiles_hs = list()
                for sample in batch:
                    smiles_hs += sample[0][1]
                smiles_hs = torch.tensor(smiles_hs, dtype=torch.float32)
                seq_feats = torch.tensor([_[1] for _ in batch], dtype=torch.int64)
                heads = torch.tensor([_[2] for _ in batch])
                tails = torch.tensor([_[3] for _ in batch])
                probs = torch.tensor([_[4] for _ in batch], dtype=torch.float32)
                uncs = torch.tensor([_[5] for _ in batch], dtype=torch.float32)
                return (smiles_gs, smiles_hs), seq_feats, heads, tails, probs, uncs

        def train_epoch_with_soft_distill(cur_model, data_loader, optimizer, args):
            cur_model.train()
            epoch_loss = 0.0
            eps = 1e-8

            for batch in data_loader:
                smiles_feats, seq_feats, heads, tails, probs, uncs = batch

                smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
                seq_feats = seq_feats.to(args.device)
                heads = heads.to(args.device)
                tails = tails.to(args.device)
                probs = probs.to(args.device)
                uncs = uncs.to(args.device)

                if isinstance(cur_model, RelationModel):
                    logits = cur_model(heads, tails)
                elif isinstance(cur_model, SelfModel):
                    logits = cur_model(smiles_feats, seq_feats)
                else:
                    logits = cur_model(heads, tails, smiles_feats, seq_feats)

                real_mask = ((uncs == 0.0) & ((probs == 0.0) | (probs == 1.0)))
                pseudo_mask = ~real_mask
                hard_labels = (probs >= 0.5).long()

                loss = 0.0
                if real_mask.any():
                    loss_real = F.cross_entropy(logits[real_mask], hard_labels[real_mask])
                    loss = loss + loss_real

                if pseudo_mask.any():
                    logits_pseudo = logits[pseudo_mask]
                    probs_pseudo = probs[pseudo_mask]
                    uncs_pseudo = uncs[pseudo_mask]
                    weights_pseudo = torch.clamp((1.0 - uncs_pseudo) ** float(args.pseudo_weight_gamma), min=0.0, max=1.0)

                    soft_targets = torch.stack([1.0 - probs_pseudo, probs_pseudo], dim=1)
                    log_probs = F.log_softmax(logits_pseudo, dim=1)
                    soft_ce = -(soft_targets * log_probs).sum(dim=1)
                    loss_pseudo = (weights_pseudo * soft_ce).sum() / (weights_pseudo.sum() + eps)

                    if real_mask.any():
                        loss = loss + args.pseudo_soft_loss_lambda * loss_pseudo
                    else:
                        loss = args.pseudo_soft_loss_lambda * loss_pseudo

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu())

            return epoch_loss
        
        train_real_triples = read_triple(os.path.join(args.data_path, 'train.tsv'), entity2id,
                                         relation2id)
        real_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id, relation2id)
        
        if fun in ("train_self", "train_relation"):
            psd_triples, psd_neg_triples, balance_times = read_psd_triples(len(train_real_triples),
                args.self_psd_use_ratio if fun=="train_relation" else args.rel_psd_use_ratio,
                args.balance_psd,
                triple_preds,
                args.self_balance_ratio if fun == "train_relation" else args.rel_balance_ratio
                )
            train_triples = [(tri, 1.0, 0.0) for tri in (train_real_triples * balance_times)]
            train_triples += [
                (tri, float(prob), float(unc))
                for tri, prob, unc in psd_triples
            ]

            train_neg_triples = [(tri, 0.0, 0.0) for tri in real_neg_triples]
            train_neg_triples += [
                (tri, float(prob), float(unc))
                for tri, prob, unc in psd_neg_triples
            ]

        elif fun == "train_projector":

            psd_triples, psd_neg_triples, balance_times = read_psd_triples(
                len(train_real_triples), args.gate_psd_use_ratio,
                args.balance_psd, triple_preds, args.gate_balance_ratio)

            train_triples = [(tri, 1.0, 0.0) for tri in (train_real_triples * balance_times)]
            train_triples += [
                (tri, float(prob), float(unc))
                for tri, prob, unc in psd_triples
            ]

            train_neg_triples = [(tri, 0.0, 0.0) for tri in real_neg_triples]
            train_neg_triples += [
                (tri, float(prob), float(unc))
                for tri, prob, unc in psd_neg_triples
            ]

        elif fun in ("train_projector_pure", "train_relation_pure", "train_self_pure"):
            if args.aug_pure_train:
                pass
            else:
                train_triples = train_real_triples
                train_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id,
                                                relation2id)
        else:
            raise ValueError
        
        valid_triples = read_triple(os.path.join(args.data_path, 'valid.tsv'), entity2id, relation2id)
        test_triples = read_triple(os.path.join(args.data_path, 'test.tsv'), entity2id, relation2id)
        valid_neg_triples = read_triple(os.path.join(args.data_path, 'valid_neg.tsv'), entity2id, relation2id)
        test_neg_triples = read_triple(os.path.join(args.data_path, 'test_neg.tsv'), entity2id, relation2id)

        if fun in ("train_self", "train_relation", "train_projector") and triple_preds is not None:
            train_weighted_dataset = WeightedSelfDataset(
                train_pos_meta=train_triples,
                train_neg_meta=train_neg_triples,
                ent_id2smiles=ent_id2smiles,
                ent_id2seq=ent_id2seq,
            )
            train_dataloader = DataLoader(
                train_weighted_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=max(1, args.cpu_num // 100),
                collate_fn=WeightedSelfDataset.collate_fn,
            )
        else:
            train_dataloader = DataLoader(SelfDataset(train_triples, train_neg_triples, ent_id2smiles, ent_id2seq),
                                          batch_size=args.train_batch_size,
                                          shuffle=True,
                                          num_workers=max(1, args.cpu_num // 100),
                                          collate_fn=SelfDataset.collate_fn
                                          )
        valid_dataloader = DataLoader(SelfDataset(valid_triples, valid_neg_triples, ent_id2smiles, ent_id2seq),
                                      batch_size=args.train_batch_size,
                                      shuffle=False,
                                      num_workers=max(1, args.cpu_num // 100),
                                      collate_fn=SelfDataset.collate_fn
                                      )
        test_dataloader = DataLoader(SelfDataset(test_triples, test_neg_triples, ent_id2smiles, ent_id2seq),
                                     batch_size=args.train_batch_size,
                                     shuffle=False,
                                     num_workers=max(1, args.cpu_num // 100),
                                     collate_fn=SelfDataset.collate_fn
                                     )

        print(f"len of train set: {len(train_triples)}")
        print(f"len of train neg set: {len(train_neg_triples)}")

        if fun in ("train_relation", "train_relation_pure"):
            if not rel_model:
                model = RelationModel(args.hidden_dim, kge_model.entity_embedding).to(args.device)
            else:
                model = rel_model


            model.freeze_embeddings()

        elif fun in ("train_self", "train_self_pure"):
            model = SelfModel(args.hidden_dim, esm_model).to(args.device)

        else:
            def count_paras(model):
                total_param = 0
                learnable_paras = 0
                print("MODEL DETAILS:\n")
                for param in model.parameters():
                    if param.requires_grad == True:
                        learnable_paras += np.prod(list(param.data.size()))
                    total_param += np.prod(list(param.data.size()))
                return total_param, learnable_paras


            rel_model.freeze_embeddings()

            model = GateModel(rel_model, self_model, args.double_layer).to(args.device)

            if not args.gate_free:
                model.freeze()


        if "train_projector" in fun and "full" not in args.data_path:
            weight_decay = args.gate_weight_decay
            learning_rate = args.gate_learning_rate
        else:
            weight_decay = args.module_weight_decay
            learning_rate = args.module_learning_rate

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        loss_fn = nn.CrossEntropyLoss()


        if "train_projector" in fun:
            es_pat = 2
        elif "train_self" in fun:
            es_pat = 5
        else:
            es_pat = 5

        es = EarlyStopping(args.model_path, args.save_opt, args.save_thre, args.save_mul, fun, patience=es_pat)

        opt_test_aupr = float("-inf")

        for i in range(max_ephs):
            if fun in ("train_self", "train_relation", "train_projector") and triple_preds is not None:
                loss = train_epoch_with_soft_distill(model, train_dataloader, optimizer, args)
            else:
                loss = model.train_epoch(train_dataloader, loss_fn, optimizer, args)
            print(f"Epoch {i} loss: {loss:>7f}")

            result = model.evaluate(valid_dataloader, args)
            acc, recall, precision, mcc, f1, auc, aupr, sensitivity, specificity = result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8]
            thred_optim = result[9]
            print(f"Valid - Accuracy: {acc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, MCC: {mcc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AUPR: {aupr:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Threshold: {thred_optim:.4f}")

            if_stop, if_opt = es(aupr, model)

            if (if_opt and "train_projector" in fun) or (if_stop and "train_projector" not in fun):
                result = model.evaluate(test_dataloader, args)
                test_acc, test_recall, test_precision, test_mcc, test_f1, test_auc, test_aupr, test_sensitivity, test_specificity = result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8]
                test_thred_optim = result[9]
                print(f"Test - Accuracy: {test_acc:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, MCC: {test_mcc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}, Sensitivity: {test_sensitivity:.4f}, Specificity: {test_specificity:.4f}, Threshold: {test_thred_optim:.4f}#")
                if test_aupr > opt_test_aupr:
                    opt_test_aupr = test_aupr


            if if_stop and max_ephs == 999999:
                break

        if args.save and "train_projector" not in fun:
            save2file(model, args.model_path, start_time, info=fun)
        current_time = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
        print(f"End training at time {current_time}")
        print(f"opt test aupr {opt_test_aupr:.4f}")

        return model, test_acc, test_auc, test_aupr
    def set_random_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dgl.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_random_seeds(args.seed)

    gc.collect()
    torch.cuda.empty_cache()
    print("Seed: ", args.seed)
    print("datasets: ", args.dataset)
    print("pseudo_uncertainty_max", args.pseudo_uncertainty_max)
    
    args.data_path = f"var_data/{args.dataset}"
    args.model_path = f"var_models/{args.dataset}"
    args.pkl_path = f"var_pkls/{args.dataset}"
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.pkl_path, exist_ok=True)
    
    args.adv = True
    args.gate_free = True
    args.balance_psd = True

    if args.baseline:
        args.load_kge_model = None
    
    print("rel_psd_use_ratio:", f"{args.rel_psd_use_ratio}")
    print("self_psd_use_ratio:", f"{args.self_psd_use_ratio}")
    print("kge_cartesian_ratio:", f"{args.kge_cartesian_ratio}")
    print("self_cartesian_ratio:", f"{args.self_cartesian_ratio}")
    print("pseudo_pos_threshold", f"{args.pseudo_pos_threshold}")
    print("pseudo_neg_threshold", f"{args.pseudo_neg_threshold}")
    print("pseudo_uncertainty_max", f"{args.pseudo_uncertainty_max}")
    if not args.baseline:
        import esm

    entity2id = dict()
    id2entity = dict()
    
    with open(os.path.join(f"{args.data_path}/../", 'entities.dict')) as fin:
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    relation2id = dict()
    id2relation = dict()
    with open(os.path.join(f"{args.data_path}/../", 'relations.dict')) as fin:
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation

    n_entity = len(entity2id)
    n_relation = len(relation2id)

    if not args.baseline:
        model_path = args.esm_model_path
        with safe_globals([argparse.Namespace]):
            esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        esm_model = esm_model.to(args.device)
        esm_converter = esm_alphabet.get_batch_converter()

        ent_id2seq, ent_id2smiles = prepare_self_esm_data(f"{args.data_path}/../ent_id2seq.csv",
                                                          f"{args.data_path}/../ent_id2smiles.csv", esm_converter)

    if args.load_kge_model:
        print("**************************load kge model******************************")
        kge_model = load_model_to_device(args.model_path, args.device, load_kge_model=args.load_kge_model)

    elif args.load_rel_model or args.load_tuned_rel_model:
        pass
    else:
        print("******************Start training kge************************")
        kge_model, kge_acc, kge_auc, kge_aupr = train_kge(args)

    if args.baseline:
        print("***********************Start train self projector******************************")
        return kge_acc, kge_auc, kge_aupr

    if args.load_rel_model:
        print("***********************load relation model******************************")
        rel_model = load_model_to_device(args.model_path, args.device, load_rel_model=args.load_rel_model)
    elif args.load_tuned_rel_model:
        pass
    else:
        print("***********************Start train relation projector******************************")
        rel_model, rel_acc, rel_auc, rel_aupr = train_module("train_relation_pure", args, kge_model=kge_model, max_ephs=args.rel_train_ephs)

    if args.kge_psd_filename:
        print("**************************load kge triple preds******************************")
        with open(f"{args.pkl_path}/{args.kge_psd_filename}", "rb") as fin:
            kge_triple_preds = pickle.load(fin)
    elif (args.load_self_model and args.gate_pure_train) or args.pure_train:
        pass
    else:
        print("***********************Start generating psd label kge************************")
        kge_triple_preds = gen_psd_label("psd_label_kge", args, rel_model,esm_converter, ent_id2seq, ent_id2smiles)


    if not args.load_self_model:
        print("************************Start training self model*****************************")
        if not args.pure_train:
            self_model, _, _, _ = train_module("train_self", args, kge_triple_preds, max_ephs=args.self_train_ephs)
        else:
            self_model, self_acc, self_auc, self_aupr = train_module("train_self_pure", args, max_ephs=args.self_train_ephs)
    else:
        print("**************************load self model******************************")
        self_model = load_model_to_device(args.model_path, args.device, load_self_model=args.load_self_model, esm_model=esm_model)
    
    if args.pure_train or (args.load_tuned_rel_model and args.gate_pure_train):
        pass
    elif not args.self_psd_filename:
        print("************************Start generating psd label self*****************************")
        self_triple_preds = gen_psd_label("psd_label_self", args, self_model, esm_converter, ent_id2seq, ent_id2smiles)
    else:
        print("**************************load self triple preds******************************")
        with open(f"{args.pkl_path}/{args.self_psd_filename}", "rb") as fin:
            self_triple_preds = pickle.load(fin)

    if args.load_tuned_rel_model:
        print("***********************load relation model******************************")
        rel_model = load_model_to_device(args.model_path, args.device, load_rel_model=args.load_tuned_rel_model)
    elif args.pure_train:
        pass
    else:
        print("***********************Start tune relation projector******************************")
        rel_model, _, _, _ = train_module("train_relation", args, self_triple_preds, rel_model=rel_model,
                                 max_ephs=args.rel_tune_ephs)

    print("***********************Start train gating model******************************")
    if not args.gate_pure_train and not args.pure_train:
        triple_preds = cut_long_preds(kge_triple_preds, self_triple_preds)
        _, _, _, _ = train_module("train_projector", args, triple_preds=triple_preds, rel_model=rel_model, self_model=self_model, max_ephs=args.gate_train_ephs)
    else:
        _, gate_acc, gate_auc, gate_aupr = train_module("train_projector_pure", args=args, rel_model=rel_model, self_model=self_model, max_ephs=args.gate_train_ephs)


    print("*************************exit*****************************")
    if args.pure_train:
        return (rel_acc, rel_auc, rel_aupr, self_acc, self_auc, self_aupr, gate_acc, gate_auc, gate_aupr)

def add_args(parser):

    parser.add_argument('--dataset', default='full_drugbank', type=str)
    parser.add_argument('--model', default='TransE', type=str)

    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    parser.add_argument('-n', '--negative_sample_size', default=256,
                        type=int)
    parser.add_argument('-d', '--hidden_dim', default=1000, type=int)
    parser.add_argument('-g', '--gamma', default=24, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('--max_relation_steps', default=999999999999999999, type=int)
    parser.add_argument('--warm_up_steps', default=999999999999999999,
                        type=int)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=50, type=int)
    parser.add_argument('--relation_valid_steps', default=11061, type=int)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0)

    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument("--relation_batch_size", default=1024, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')
    parser.add_argument("--load_kge_model", type=str, default=None,
                        help="KGE checkpoint filename in var_models/")
    parser.add_argument("--esm_model_path", type=str,
                        default="esm/esm1b_t33_650M_UR50S.pt",
                        help="Path to the ESM-1b checkpoint")
    parser.add_argument("--load_self_model", type=str)
    parser.add_argument("--load_rel_model", type=str)
    parser.add_argument("--load_tuned_rel_model", type=str)
    parser.add_argument("--kge_psd_filename", type=str)
    parser.add_argument("--self_psd_filename", type=str)
    parser.add_argument("--gate_free", action='store_true')
    parser.add_argument("--balance_psd", action="store_true")
    parser.add_argument("--double_layer", action="store_true")
    parser.add_argument("--save_mul", action="store_true")
    parser.add_argument("--save_opt", action="store_true")
    parser.add_argument("--save_thre", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--rel_psd_use_ratio", type=float,
                        default=0.00003)
    parser.add_argument("--self_psd_use_ratio", type=float,
                        default=0.0003)
    parser.add_argument("--gate_psd_use_ratio", type=float, default=0.003)
    parser.add_argument("--kge_cartesian_ratio", type=float, default=0.02)
    parser.add_argument("--self_cartesian_ratio", type=float, default=0.0002)
    parser.add_argument('--save', action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--pure_train", action="store_true")
    parser.add_argument("--gate_pure_train", action="store_true")
    parser.add_argument("--rel_train_ephs", type=int, default=999999)
    parser.add_argument("--self_train_ephs", type=int, default=999999)
    parser.add_argument("--rel_tune_ephs", type=int, default=999999)
    parser.add_argument("--gate_train_ephs", type=int, default=999999)
    parser.add_argument("--self_balance_ratio", type=float, default=1.0)
    parser.add_argument("--rel_balance_ratio", type=float, default=0.3)
    parser.add_argument("--gate_balance_ratio", type=float, default=1.0)
    parser.add_argument("--mc_dropout_passes", type=int, default=5)
    
    parser.add_argument("--pseudo_pos_threshold", type=float, default=0.7)
    parser.add_argument("--pseudo_neg_threshold", type=float, default=0.3)
    parser.add_argument("--pseudo_uncertainty_max", type=float, default=0.3)
    parser.add_argument("--prototype_temperature", type=float, default=0.5)
    parser.add_argument("--pseudo_topk_per_class", type=int, default=100)
    parser.add_argument("--pseudo_soft_loss_lambda", type=float, default=1.0)
    parser.add_argument("--pseudo_weight_gamma", type=float, default=1.0)

    parser.add_argument("--module_weight_decay", type=float, default=0)
    parser.add_argument("--gate_weight_decay", type=float, default=1e-2)
    parser.add_argument('--module_learning_rate', default=1e-3, type=float)
    parser.add_argument('--gate_learning_rate', default=1e-2, type=float)
    parser.add_argument('--aug_pure_train', action="store_true")
    return parser


if __name__ == '__main__':
    start_time = str(datetime.datetime.now())[0:-4].replace(' ', '_').replace(":", "_")

    parser = argparse.ArgumentParser()

    parser = add_args(parser)
    args = parser.parse_args()
    args.device = int(args.device)
    main(args, start_time)
