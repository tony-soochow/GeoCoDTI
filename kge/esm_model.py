from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nnfun
from torch.nn.utils import spectral_norm
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgllife.model import MLPNodeReadout
from dgl import function as fn

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve, confusion_matrix, matthews_corrcoef, precision_score, recall_score

def _sn(module, use_spectral_norm=False):
    return spectral_norm(module) if use_spectral_norm else module

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.kwargs = {"model_name": model_name, "nentity":nentity, "nrelation":nrelation,
                       "hidden_dim": hidden_dim, "gamma":gamma,
                       "double_entity_embedding": double_entity_embedding,
                       "double_relation_embedding": double_relation_embedding
                       }
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.tensor([[0.5 * self.embedding_range.item()]]))

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')






    def forward(self, sample, module_name, mode='single'):
        if mode in ["single", "single-head-batch", "single-tail-batch"]:
            if module_name in ["molecule_train", "protein_train", "fuse_test", "molecule_test"]:
                sample, self_data = sample
            elif module_name == "relation":
                pass
            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            if module_name == "fuse_test":
                head_smiles_batch = self.encode_compounds(self_data["head_smiles_batch"]).unsqueeze(1)
                tail_smiles_batch = self.encode_compounds(self_data["tail_smiles_batch"]).unsqueeze(1)
                head_seq_batch = self.encode_proteins(self_data["head_seq_batch"]).unsqueeze(1)
                tail_seq_batch = self.encode_proteins(self_data["tail_seq_batch"]).unsqueeze(1)

                head[self_data["if_head_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((head[self_data["if_head_smiles_batch"]], head_smiles_batch), dim=2))
                tail[self_data["if_tail_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((tail[self_data["if_tail_smiles_batch"]], tail_smiles_batch), dim=2))
                head[self_data["if_head_seq_batch"]] = self.protein_mlp(
                    torch.cat((head[self_data["if_head_seq_batch"]], head_seq_batch), dim=2))
                tail[self_data["if_tail_seq_batch"]] = self.protein_mlp(
                    torch.cat((tail[self_data["if_tail_seq_batch"]], tail_seq_batch), dim=2))


            elif module_name == "molecule_test":
                head_smiles_batch = self.encode_compounds(self_data["head_smiles_batch"]).unsqueeze(1)
                tail_smiles_batch = self.encode_compounds(self_data["tail_smiles_batch"]).unsqueeze(1)
                head[self_data["if_head_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((head[self_data["if_head_smiles_batch"]], head_smiles_batch), dim=2))
                tail[self_data["if_tail_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((tail[self_data["if_tail_smiles_batch"]], tail_smiles_batch), dim=2))


            elif module_name == "molecule_train":
                if mode == "single-head-batch":
                    self_tail = self.encode_compounds(self_data).unsqueeze(1)
                    tail = self.molecule_mlp(torch.cat((self_tail, tail), 2))
                elif mode == "single-tail-batch":
                    self_head = self.encode_compounds(self_data).unsqueeze(1)
                    head = self.molecule_mlp(torch.cat((self_head, head), 2))
                else:
                    raise ValueError('mode %s not supported' % mode)


            elif module_name == "protein_train":
                if mode == "single-head-batch":
                    self_tail = self.encode_proteins(self_data).unsqueeze(1)
                    tail = self.protein_mlp(torch.cat((self_tail, tail), 2))
                elif mode == "single-tail-batch":
                    self_head = self.encode_proteins(self_data).unsqueeze(1)
                    head = self.protein_mlp(torch.cat((self_head, head), 2))
                else:
                    raise ValueError('mode %s not supported' % mode)




        elif mode == "head-batch":
            if module_name == "relation":
                tail_part, head_part = sample


                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)

            elif module_name == "molecule_train":
                tail_part, head_part, smiles_batch = sample
                tail = self.encode_compounds(smiles_batch).unsqueeze(1)
                tail_free = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)
                tail = self.molecule_mlp(torch.cat((tail, tail_free), 2))

            elif module_name == "protein_train":
                tail_part, head_part, seq_batch = sample
                tail = self.encode_proteins(seq_batch).unsqueeze(1)
                tail_free = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)
                tail = self.protein_mlp(torch.cat((tail, tail_free), 2))

            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")


            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)


        elif mode == "tail-batch":
            if module_name == "relation":
                head_part, tail_part = sample

                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 2]
                ).unsqueeze(1)

            elif module_name == "molecule_train":
                head_part, tail_part, smiles_batch = sample
                head = self.encode_compounds(smiles_batch).unsqueeze(1)
                head_free = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 2]
                ).unsqueeze(1)
                head = self.molecule_mlp(torch.cat((head, head_free), 2))

            elif module_name == "protein_train":
                head_part, tail_part, seq_batch = sample
                head = self.encode_proteins(seq_batch).unsqueeze(1)
                head_free = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 2]
                ).unsqueeze(1)
                head = self.protein_mlp(torch.cat((head, head_free), 2))

            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == "single-head-batch":
            sample, self_feat = sample
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            if module_name == "molecule_train":
                tail = self.encode_compounds(self_feat).unsqueeze(1)
            elif module_name == "protein_train":
                tail = self.encode_proteins(self_feat).unsqueeze(1)
            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")


        elif mode == "single-tail-batch":
            sample, self_feat = sample
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
            if module_name == "molecule_train":
                head = self.encode_compounds(self_feat).unsqueeze(1)
            elif module_name == "protein_train":
                head = self.encode_proteins(self_feat).unsqueeze(1)
            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")

        else:
            raise ValueError('mode %s not supported' % mode)


        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)


        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846


        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def kge_loss(model, optimizer, positive_score, negative_score, subsampling_weight, args):
        if args.negative_adversarial_sampling:
            negative_score = (nnfun.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * nnfun.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = nnfun.logsigmoid(-negative_score).mean(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss


    @staticmethod
    def train_relation_step(model, optimizer, train_iterator, args):

        model.train()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        positive_sample = positive_sample.to(args.device)
        negative_sample = negative_sample.to(args.device)
        subsampling_weight = subsampling_weight.to(args.device)

        negative_score = model((positive_sample, negative_sample), module_name="relation", mode=mode)

        positive_score = model(positive_sample, module_name="relation", mode="single")

        positive_score = nnfun.logsigmoid(positive_score).squeeze(dim=1)


        return model.kge_loss(model, optimizer, positive_score, negative_score, subsampling_weight, args)


    @staticmethod
    def evaluate(args, model, test_dataloader, labels, if_metric=True):
        def correct_count(labels, scores):
            return torch.isclose(scores, labels, atol=0.5, rtol=0).sum().item()



        model.eval()
        preds = list()
        with torch.no_grad():
            for i, sample in enumerate(test_dataloader):
                sample = sample.to(args.device)

                scores = torch.sigmoid(model(sample, module_name="relation", mode="single"))
                scores = scores.cpu().squeeze()
                preds += scores.tolist()
                if i % 10000 == 0 and not if_metric:
                    print(f"batch {i} finished")

        if if_metric:
            acc = correct_count(labels, torch.tensor(preds)) / len(test_dataloader.dataset)
            auc = roc_auc_score(labels.numpy(), preds)
            aupr = average_precision_score(labels.numpy(), preds)
            return preds, acc, auc, aupr
        else:
            return preds


class SelfModel(nn.Module):
    def __init__(self, hidden_dim, esm_model, use_spectral_norm=False):
        super().__init__()
        self.kwargs = {"hidden_dim": hidden_dim, "use_spectral_norm": use_spectral_norm}
        self.entity_dim = hidden_dim
        self.dropout_p = 0.2
        self.layer_filters_proteins = [1280, 96, 128, self.entity_dim]
        self.cpi_hidden_dim = [78, self.entity_dim, self.entity_dim]

        self.drug_gcn = nn.ModuleList(
            [GraphConv(in_feats=self.cpi_hidden_dim[i], out_feats=self.cpi_hidden_dim[i + 1]) for i in
             range(len(self.cpi_hidden_dim) - 1)])
        self.drug_output_layer = MLPNodeReadout(self.entity_dim, self.entity_dim, self.entity_dim,
                                                activation=nn.ReLU(), mode='max')

        self.target_cnn = nn.ModuleList(
            [_sn(nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i + 1],
                       kernel_size=3, padding=1), use_spectral_norm) for i in range(len(self.layer_filters_proteins) - 1)])

        self.fc = nn.Sequential(
            _sn(nn.Linear(self.entity_dim * 2, self.entity_dim), use_spectral_norm),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            _sn(nn.Linear(self.entity_dim, 2), use_spectral_norm)
        )
        self.esm_model = esm_model
        for para in self.esm_model.parameters():
            para.requires_grad = False




    def encode_compounds(self, smiles_batch):
        compound_graphs, compound_vectors = smiles_batch

        for l in self.drug_gcn:
            compound_vectors = nnfun.relu(l(compound_graphs,
                                            compound_vectors))

        compound_vectors = nnfun.relu(self.drug_output_layer(compound_graphs, compound_vectors.squeeze()))

        return compound_vectors



    def encode_proteins(self, protein_vectors):
        protein_vectors = self.esm_model(protein_vectors, repr_layers=[33], return_contacts=False)
        protein_vectors = protein_vectors["representations"][33]
        protein_vectors = protein_vectors.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vectors = nnfun.relu(l(protein_vectors))
        protein_vectors = nnfun.adaptive_max_pool1d(protein_vectors, output_size=1)
        protein_vectors = protein_vectors.view(protein_vectors.size(0), -1)

        return protein_vectors


    def forward(self, smiles_embeds, seq_embeds, return_features=False):
        smiles_embeds = self.encode_compounds(smiles_embeds)
        seq_embeds = self.encode_proteins(seq_embeds)
        fused_features = torch.cat((smiles_embeds, seq_embeds), 1)
        res = self.fc(fused_features)
        
        if return_features:
            return res, fused_features
        return res


    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, (smiles_feats, seq_feats, labels, _, _) in enumerate(data_loader):
            smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
            seq_feats = seq_feats.to(args.device)
            labels = labels.to(args.device)
            outs = self(smiles_feats, seq_feats)
            loss = loss_fn(outs, labels)
            epoch_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss

    def evaluate(self, data_loader, args, if_metric=True):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        with torch.no_grad():
            for i, (smiles_feats, seq_feats, labels, _, _) in enumerate(data_loader):
                smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
                seq_feats = seq_feats.to(args.device)
                out = self(smiles_feats, seq_feats)
                all_probs += nnfun.softmax(out, dim=1)[:, 1].cpu().tolist()
                all_classes += out.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished")

        if if_metric:
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            
            y_pred = np.array(all_probs)
            y_label = np.array(all_labels)
            
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec_curve, recall_curve, _ = precision_recall_curve(y_label, y_pred)

            precision_tpr = tpr / (tpr + fpr + 0.00001)
            f1 = 2 * precision_tpr * tpr / (tpr + precision_tpr + 0.00001)

            thred_optim = thresholds[5:][np.argmax(f1[5:])] if len(thresholds) > 5 else thresholds[np.argmax(f1)]
            thred_optim = float(thred_optim)

            y_pred_binary = [1 if i >= thred_optim else 0 for i in y_pred]

            cm = confusion_matrix(y_label, y_pred_binary)
            acc = accuracy_score(y_label, y_pred_binary)
            recall = recall_score(y_label, y_pred_binary, zero_division=0)

            sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            mcc = matthews_corrcoef(y_label, y_pred_binary)
            precision = precision_score(y_label, y_pred_binary, zero_division=0)
            f1_max = np.max(f1[5:]) if len(f1) > 5 else np.max(f1)
            
            return acc, recall, precision, mcc, f1_max, auc, aupr, sensitivity, specificity, thred_optim, cm, y_pred, fpr, tpr
        else:
            return all_probs



class SelfFAModel(nn.Module):
    def __init__(self, hidden_dim, esm_model, use_spectral_norm=False):

        class FALayer(nn.Module):
            def __init__(self, in_dim, out_dim, dropout, subspace_num, save_model, use_spectral_norm=False):
                super(FALayer, self).__init__()
                self.dropout_list = nn.ModuleList()
                self.project = _sn(nn.Linear(in_dim, out_dim, bias=True), use_spectral_norm)
                self.save_model = save_model
                self.out_dim = out_dim
                self.subspace_num = subspace_num
                self.dim_per_space = out_dim // subspace_num

                self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, self.subspace_num, self.dim_per_space)))
                self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, self.subspace_num, self.dim_per_space)))
                self.dropout = nn.Dropout(dropout)

                self.save_alphas = False
                self.alphas = list()

                self.reset_parameters()

            def reset_parameters(self):
                gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_normal_(self.project.weight, gain=gain)
                nn.init.xavier_normal_(self.attn_l, gain=gain)
                nn.init.xavier_normal_(self.attn_r, gain=gain)

            def degree_mul(self, edges):
                alpha = torch.tanh(edges.data['e'])
                if self.save_alphas:
                    self.alphas.append(alpha)
                alpha_norm = edges.dst['d'].view(-1, 1, 1) * edges.src['d'].view(-1, 1, 1) * alpha
                alpha_norm = self.dropout(alpha_norm)
                edges.data.pop('e')
                return {'alpha_norm': alpha_norm}

            def forward(self, g, h):
                h_projected = self.project(h).view(-1, self.subspace_num, self.dim_per_space)
                el = (h_projected * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (h_projected * self.attn_r).sum(dim=-1).unsqueeze(-1)
                g.ndata.update({'ft': h_projected, 'el': el, 'er': er})
                g.apply_edges(fn.u_add_v('el', 'er', 'e'))
                g.ndata.pop('el')
                g.ndata.pop('er')
                g.apply_edges(self.degree_mul)
                g.update_all(fn.u_mul_e('ft', 'alpha_norm', 'm'),
                             fn.sum('m', 'ft'))
                g.edata.pop('alpha_norm')
                ft = g.ndata.pop('ft')
                ft = ft.view(-1, self.out_dim)
                return ft

        super().__init__()
        self.kwargs = {"hidden_dim": hidden_dim, "use_spectral_norm": use_spectral_norm}
        self.entity_dim = hidden_dim
        self.layer_filters_proteins = [1280, 96, 128, self.entity_dim]
        self.cpi_hidden_dim = [78, self.entity_dim, self.entity_dim]

        self.drug_gcn = nn.ModuleList(
            [FALayer(self.cpi_hidden_dim[i], self.cpi_hidden_dim[i + 1], 0, 1, False, use_spectral_norm=use_spectral_norm) for i in
             range(len(self.cpi_hidden_dim) - 1)])
        self.drug_output_layer = MLPNodeReadout(self.entity_dim, self.entity_dim, self.entity_dim,
                                                activation=nn.ReLU(), mode='max')

        self.target_cnn = nn.ModuleList(
            [_sn(nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i + 1],
                       kernel_size=3, padding=1), use_spectral_norm) for i in range(len(self.layer_filters_proteins) - 1)])

        self.fc = nn.Sequential(_sn(nn.Linear(self.entity_dim*2, self.entity_dim), use_spectral_norm), nn.ReLU(), _sn(nn.Linear(self.entity_dim, 2), use_spectral_norm))
        self.esm_model = esm_model
        for para in self.esm_model.parameters():
            para.requires_grad = False




    def encode_compounds(self, smiles_batch):
        compound_graphs, compound_vectors = smiles_batch
        if len(compound_vectors) > 0:

            for l in self.drug_gcn:
                compound_vectors = nnfun.relu(l(compound_graphs,
                                                compound_vectors))

            compound_vectors = nnfun.relu(self.drug_output_layer(compound_graphs, compound_vectors.squeeze()))
        else:
            compound_vectors = torch.zeros(0, self.entity_dim).to(self.device)

        return compound_vectors



    def encode_proteins(self, protein_vectors):
        protein_vectors = self.esm_model(protein_vectors, repr_layers=[33], return_contacts=False)
        protein_vectors = protein_vectors["representations"][33]
        protein_vectors = protein_vectors.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vectors = nnfun.relu(l(protein_vectors))
        protein_vectors = nnfun.adaptive_max_pool1d(protein_vectors, output_size=1)
        protein_vectors = protein_vectors.view(protein_vectors.size(0), -1)

        return protein_vectors



    def forward(self, smiles_embeds, seq_embeds, return_features=False):
        smiles_embeds = self.encode_compounds(smiles_embeds)
        seq_embeds = self.encode_proteins(seq_embeds)
        fused_features = torch.cat((smiles_embeds, seq_embeds), 1)
        res = self.fc(fused_features)
        
        if return_features:
            return res, fused_features
        return res


    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, (smiles_feats, seq_feats, labels, _, _) in enumerate(data_loader):
            smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
            seq_feats = seq_feats.to(args.device)
            labels = labels.to(args.device)
            outs = self(smiles_feats, seq_feats)
            loss = loss_fn(outs, labels)
            epoch_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss

    def evaluate(self, data_loader, args, if_metric=True):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        with torch.no_grad():
            for i, (smiles_feats, seq_feats, labels, _, _) in enumerate(data_loader):
                smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
                seq_feats = seq_feats.to(args.device)
                out = self(smiles_feats, seq_feats)
                all_probs += nnfun.softmax(out, dim=1)[:, 1].cpu().tolist()
                all_classes += out.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished")

        if if_metric:
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            
            y_pred = np.array(all_probs)
            y_label = np.array(all_labels)
            
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec_curve, recall_curve, _ = precision_recall_curve(y_label, y_pred)

            precision_tpr = tpr / (tpr + fpr + 0.00001)
            f1 = 2 * precision_tpr * tpr / (tpr + precision_tpr + 0.00001)

            thred_optim = thresholds[5:][np.argmax(f1[5:])] if len(thresholds) > 5 else thresholds[np.argmax(f1)]
            thred_optim = float(thred_optim)

            y_pred_binary = [1 if i >= thred_optim else 0 for i in y_pred]

            cm = confusion_matrix(y_label, y_pred_binary)
            acc = accuracy_score(y_label, y_pred_binary)
            recall = recall_score(y_label, y_pred_binary, zero_division=0)

            sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            mcc = matthews_corrcoef(y_label, y_pred_binary)
            precision = precision_score(y_label, y_pred_binary, zero_division=0)
            f1_max = np.max(f1[5:]) if len(f1) > 5 else np.max(f1)
            
            return acc, recall, precision, mcc, f1_max, auc, aupr, sensitivity, specificity, thred_optim, cm, y_pred, fpr, tpr
        else:
            return all_probs




class RelationModel(nn.Module):
    def __init__(self, entity_dim, entity_embeddings, use_spectral_norm=False):
        super().__init__()
        self.dropout_p = 0.2
        self.fc = nn.Sequential(
            _sn(nn.Linear(entity_dim * 2, entity_dim), use_spectral_norm),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            _sn(nn.Linear(entity_dim, 2), use_spectral_norm)
        )
        self.entity_embeddings = nn.Parameter(entity_embeddings)
        self.kwargs = {
            "entity_dim": entity_dim,
            "use_spectral_norm": use_spectral_norm,
        }

    def forward(self, heads, tails, return_features=False):
        heads = self.entity_embeddings[heads]
        tails = self.entity_embeddings[tails]
        fused_features = torch.cat((heads, tails), 1)
        res = self.fc(fused_features)
        
        if return_features:
            return res, fused_features
        return res

    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, (_, _, labels, heads, tails) in enumerate(data_loader):
            heads = heads.to(args.device)
            tails = tails.to(args.device)
            labels = labels.to(args.device)
            outs = self(heads, tails)
            loss = loss_fn(outs, labels)
            epoch_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss


    def evaluate(self, data_loader, args, if_metric=True):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        with torch.no_grad():
            for i, (_, _, labels, heads, tails) in enumerate(data_loader):
                heads = heads.to(args.device)
                tails = tails.to(args.device)
                outs = self(heads, tails)
                all_probs += nnfun.softmax(outs, dim=1)[:, 1].cpu().tolist()
                all_classes += outs.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished")

        if if_metric:
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            
            y_pred = np.array(all_probs)
            y_label = np.array(all_labels)
            
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec_curve, recall_curve, _ = precision_recall_curve(y_label, y_pred)

            precision_tpr = tpr / (tpr + fpr + 0.00001)
            f1 = 2 * precision_tpr * tpr / (tpr + precision_tpr + 0.00001)

            thred_optim = thresholds[5:][np.argmax(f1[5:])] if len(thresholds) > 5 else thresholds[np.argmax(f1)]
            thred_optim = float(thred_optim)

            y_pred_binary = [1 if i >= thred_optim else 0 for i in y_pred]

            cm = confusion_matrix(y_label, y_pred_binary)
            acc = accuracy_score(y_label, y_pred_binary)
            recall = recall_score(y_label, y_pred_binary, zero_division=0)
            sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            mcc = matthews_corrcoef(y_label, y_pred_binary)
            precision = precision_score(y_label, y_pred_binary, zero_division=0)
            f1_max = np.max(f1[5:]) if len(f1) > 5 else np.max(f1)
            
            return acc, recall, precision, mcc, f1_max, auc, aupr, sensitivity, specificity, thred_optim, cm, y_pred, fpr, tpr
        else:
            return all_probs


    def freeze_embeddings(self):
        self.entity_embeddings.requires_grad = False

    def unfreeze_embeddings(self):
        self.entity_embeddings.requires_grad = True


class GateModel(nn.Module):
    def __init__(self, relation_model, self_model, if_double_layer, use_spectral_norm=False):
        super().__init__()
        self.relation_model = relation_model
        self.self_model = self_model
        self.kwargs = {"if_double_layer": if_double_layer, "use_spectral_norm": use_spectral_norm}
        hid_dim = self.self_model.entity_dim
        if if_double_layer:
            self.fc = nn.Sequential(_sn(nn.Linear(hid_dim*4, hid_dim), use_spectral_norm), nn.ReLU(),
                                    _sn(nn.Linear(hid_dim, 2), use_spectral_norm))
        else:
            self.fc = _sn(nn.Linear(hid_dim*4, 2), use_spectral_norm)
        self.relation_model.freeze_embeddings()


    def freeze(self):
        for para in self.relation_model.parameters():
            para.requires_grad = False
        for para in self.self_model.parameters():
            para.requires_grad = False

    def forward(self, heads, tails, smiles_embeds, seq_embeds):
        smiles_embeds = self.self_model.encode_compounds(smiles_embeds)
        seq_embeds = self.self_model.encode_proteins(seq_embeds)
        heads = self.relation_model.entity_embeddings[heads]
        tails = self.relation_model.entity_embeddings[tails]
        weight = self.fc(torch.cat((heads, tails, smiles_embeds, seq_embeds), 1)).softmax(1)
        res_self = self.self_model.fc(torch.cat((smiles_embeds, seq_embeds), 1))
        res_relation = self.relation_model.fc(torch.cat((heads, tails), 1))
        res = weight[:,0].unsqueeze(1) * res_self + weight[:,1].unsqueeze(1) * res_relation

        return res


    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, (smiles_feats, seq_feats, labels, heads, tails) in enumerate(data_loader):
            smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
            seq_feats = seq_feats.to(args.device)
            heads = heads.to(args.device)
            tails = tails.to(args.device)
            labels = labels.to(args.device)
            outs = self(heads, tails, smiles_feats, seq_feats)
            loss = loss_fn(outs, labels)
            epoch_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss

    def evaluate(self, data_loader, args, if_metric=True, if_probs=False):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        with torch.no_grad():
            for i, (smiles_feats, seq_feats, labels, heads, tails) in enumerate(data_loader):
                smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
                seq_feats = seq_feats.to(args.device)
                heads = heads.to(args.device)
                tails = tails.to(args.device)
                outs = self(heads, tails, smiles_feats, seq_feats)
                if if_probs:
                    print(f"batch {i}:", end=" ")
                    print(nnfun.softmax(outs, dim=1)[:, 1].cpu().tolist())
                all_probs += nnfun.softmax(outs, dim=1)[:, 1].cpu().tolist()
                all_classes += outs.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished")

        if if_metric:
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            
            y_pred = np.array(all_probs)
            y_label = np.array(all_labels)
            
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec_curve, recall_curve, _ = precision_recall_curve(y_label, y_pred)

            precision_tpr = tpr / (tpr + fpr + 0.00001)
            f1 = 2 * precision_tpr * tpr / (tpr + precision_tpr + 0.00001)

            thred_optim = thresholds[5:][np.argmax(f1[5:])] if len(thresholds) > 5 else thresholds[np.argmax(f1)]
            thred_optim = float(thred_optim)

            y_pred_binary = [1 if i >= thred_optim else 0 for i in y_pred]

            cm = confusion_matrix(y_label, y_pred_binary)
            acc = accuracy_score(y_label, y_pred_binary)
            recall = recall_score(y_label, y_pred_binary, zero_division=0)

            sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            mcc = matthews_corrcoef(y_label, y_pred_binary)
            precision = precision_score(y_label, y_pred_binary, zero_division=0)
            f1_max = np.max(f1[5:]) if len(f1) > 5 else np.max(f1)
            
            return acc, recall, precision, mcc, f1_max, auc, aupr, sensitivity, specificity, thred_optim, cm, y_pred, fpr, tpr
        else:
            return all_probs


    def evaluate_experts(self, data_loader, args, if_metric=True):
        self.eval()
        self_all_probs, self_all_classes, all_labels = list(), list(), list()
        rel_all_probs, rel_all_classes = list(), list()
        with torch.no_grad():
            for i, (smiles_feats, seq_feats, labels, heads, tails) in enumerate(data_loader):
                smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
                seq_feats = seq_feats.to(args.device)
                heads = heads.to(args.device)
                tails = tails.to(args.device)
                self_outs = self.self_model(smiles_feats, seq_feats)
                rel_outs = self.relation_model(heads, tails)
                self_all_probs += nnfun.softmax(self_outs, dim=1)[:, 1].cpu().tolist()
                rel_all_probs += nnfun.softmax(rel_outs, dim=1)[:, 1].cpu().tolist()
                self_all_classes += self_outs.argmax(1).cpu().tolist()
                rel_all_classes += rel_outs.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished")

        if if_metric:
            y_label = np.array(all_labels)
            
            self_auc = roc_auc_score(all_labels, self_all_probs)
            self_aupr = average_precision_score(all_labels, self_all_probs)
            
            self_y_pred = np.array(self_all_probs)
            self_fpr, self_tpr, self_thresholds = roc_curve(y_label, self_y_pred)
            self_prec, self_recall_curve, _ = precision_recall_curve(y_label, self_y_pred)
            self_precision_tpr = self_tpr / (self_tpr + self_fpr + 0.00001)
            self_f1 = 2 * self_precision_tpr * self_tpr / (self_tpr + self_precision_tpr + 0.00001)
            self_thred_optim = self_thresholds[5:][np.argmax(self_f1[5:])] if len(self_thresholds) > 5 else self_thresholds[np.argmax(self_f1)]
            self_y_pred_binary = [1 if i >= self_thred_optim else 0 for i in self_y_pred]
            self_cm = confusion_matrix(y_label, self_y_pred_binary)
            self_acc = accuracy_score(y_label, self_y_pred_binary)
            self_recall = recall_score(y_label, self_y_pred_binary, zero_division=0)
            self_sensitivity = self_cm[1, 1] / (self_cm[1, 0] + self_cm[1, 1]) if (self_cm[1, 0] + self_cm[1, 1]) > 0 else 0
            self_specificity = self_cm[0, 0] / (self_cm[0, 0] + self_cm[0, 1]) if (self_cm[0, 0] + self_cm[0, 1]) > 0 else 0
            self_mcc = matthews_corrcoef(y_label, self_y_pred_binary)
            self_precision = precision_score(y_label, self_y_pred_binary, zero_division=0)
            self_f1_max = np.max(self_f1[5:]) if len(self_f1) > 5 else np.max(self_f1)
            
            rel_auc = roc_auc_score(all_labels, rel_all_probs)
            rel_aupr = average_precision_score(all_labels, rel_all_probs)
            
            rel_y_pred = np.array(rel_all_probs)
            rel_fpr, rel_tpr, rel_thresholds = roc_curve(y_label, rel_y_pred)
            rel_prec, rel_recall_curve, _ = precision_recall_curve(y_label, rel_y_pred)
            rel_precision_tpr = rel_tpr / (rel_tpr + rel_fpr + 0.00001)
            rel_f1 = 2 * rel_precision_tpr * rel_tpr / (rel_tpr + rel_precision_tpr + 0.00001)
            rel_thred_optim = rel_thresholds[5:][np.argmax(rel_f1[5:])] if len(rel_thresholds) > 5 else rel_thresholds[np.argmax(rel_f1)]
            rel_y_pred_binary = [1 if i >= rel_thred_optim else 0 for i in rel_y_pred]
            rel_cm = confusion_matrix(y_label, rel_y_pred_binary)
            rel_acc = accuracy_score(y_label, rel_y_pred_binary)
            rel_recall = recall_score(y_label, rel_y_pred_binary, zero_division=0)
            rel_sensitivity = rel_cm[1, 1] / (rel_cm[1, 0] + rel_cm[1, 1]) if (rel_cm[1, 0] + rel_cm[1, 1]) > 0 else 0
            rel_specificity = rel_cm[0, 0] / (rel_cm[0, 0] + rel_cm[0, 1]) if (rel_cm[0, 0] + rel_cm[0, 1]) > 0 else 0
            rel_mcc = matthews_corrcoef(y_label, rel_y_pred_binary)
            rel_precision = precision_score(y_label, rel_y_pred_binary, zero_division=0)
            rel_f1_max = np.max(rel_f1[5:]) if len(rel_f1) > 5 else np.max(rel_f1)
            
            return (self_acc, self_recall, self_precision, self_mcc, self_f1_max, self_auc, self_aupr, self_sensitivity, self_specificity,  
                    rel_acc, rel_recall, rel_precision, rel_mcc, rel_f1_max, rel_auc, rel_aupr, rel_sensitivity, rel_specificity)
        else:
            return self_all_probs, rel_all_probs