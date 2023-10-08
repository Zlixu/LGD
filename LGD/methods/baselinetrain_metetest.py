import backbone
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BaselineTrain(nn.Module):
    def __init__(self, params, model_func, num_class, loss_type='softmax'):
        super(BaselineTrain, self).__init__()
        self.params = params
        self.feature = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
        self.loss_type = loss_type  # 'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        if params.st_align or params.ts_align:
            self.threshold = params.threshold
            self.teacher_feature = copy.deepcopy(self.feature)
            self.teacher_classifier = copy.deepcopy(self.classifier)
            self.init_teacher()
        if params.st_align:
            self.register_buffer('target_proto', torch.zeros(self.num_class, self.feature.final_feat_dim))
            self.iter_num = 0
            self.max_iter = 50000
            self.alpha = 1.
            self.st_align_lw = 0.
            self.st_align_tau = nn.Parameter(torch.tensor(1.)*params.st_align_tau)
        if params.ts_align:
            self.ts_align_tau = nn.Parameter(torch.tensor(1.)*params.ts_align_tau)

    def l2distance(self,x, y):
        assert x.shape[:-2] == y.shape[:-2]
        prefix_shape = x.shape[:-2]

        c, M_x = x.shape[-2:]
        M_y = y.shape[-1]
    
        x = x.view(-1, c, M_x)
        y = y.view(-1, c, M_y)
    
        x_t = x.transpose(1, 2)
        x_t2 = x_t.pow(2.0).sum(-1, keepdim=True)
        y2 = y.pow(2.0).sum(1, keepdim=True)

        ret = x_t2 + y2 - 2.0 * x_t@y
        ret = ret.view(prefix_shape + (M_x, M_y))
        return ret


    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def init_teacher(self):
        for param_t, param_s in zip(self.teacher_feature.state_dict().values(), self.feature.state_dict().values()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        for param_t, param_s in zip(self.teacher_classifier.state_dict().values(), self.classifier.state_dict().values()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

    def forward(self, x):
        x = x.cuda()
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def teacher_forward(self, x):
        x = x.cuda()
        x = self.teacher_feature(x)
        x = self.teacher_classifier(x)
        x = F.softmax(x, dim=-1)
        max_prob, pred = x.max(dim=-1)
        return max_prob.cpu(), pred.cpu(), x.cpu()

    def forward_loss(self, x, y):
        x = x.cuda()
        feature = self.feature(x)
        scores = self.classifier(feature)
        y = y.cuda()
        if len(y.shape) > 1:
            pred = F.log_softmax(scores, -1)
            loss = F.kl_div(pred, y, reduction='batchmean')
        else:
            loss = self.loss_fn(scores, y)
        return loss, feature

    def get_pseudo_loader(self, unlabeled_loader, soft_label=False, threshold=0, num_img=1):
        with torch.no_grad():
            selected_y = []
            selected_idx = []
            for x, _, idx in unlabeled_loader:
                if num_img > 1:
                    x = x[0]
                confidence, pred, prob = self.teacher_forward(x)
                selected_idx.append(idx[confidence > threshold])
                if soft_label:
                    selected_y.append(prob[confidence > threshold])
                else:
                    selected_y.append(pred[confidence > threshold])
            selected_idx = torch.cat(selected_idx).detach().cpu().numpy()
            selected_y = torch.cat(selected_y).detach().cpu()

            class NewDataset:
                def __init__(self, dataset, label):
                    self.dataset = dataset
                    self.label = label

                def __getitem__(self, index):
                    data, *_ = self.dataset[index]
                    label = self.label[index]
                    return data, label

                def __len__(self):
                    return len(self.dataset)

            if len(selected_idx) > 0:
                n_pseudo = len(selected_idx)
                n_total = len(unlabeled_loader.dataset)
                print(f'Select {n_pseudo} ({100.0 * n_pseudo / n_total:.2f}%) pesudo samples')
                pseudo_dataset = torch.utils.data.Subset(unlabeled_loader.dataset, selected_idx)
                pseudo_dataset = NewDataset(pseudo_dataset, selected_y)
                new_loader =  torch.utils.data.DataLoader(pseudo_dataset,
                                                           batch_size=unlabeled_loader.batch_size,
                                                           shuffle=True,
                                                           num_workers=12,
                                                           drop_last=True)
                return new_loader

    def update_target_proto(self, fux, pseudo_label):
        with torch.no_grad():
            protos = [0 for i in range(self.num_class)]
            sample_num = [0 for i in range(self.num_class)]
            for i, cls in enumerate(pseudo_label.cpu().numpy()):
                protos[cls] += fux[i]
                sample_num[cls] += 1
            for i in range(self.num_class):
                if sample_num[i] > 0:
                    protos[i] /= sample_num[i]
                    self.target_proto[i] = self.params.prototype_m * self.target_proto[i] +\
                                           (1-self.params.prototype_m) * protos[i]
        self.iter_num += 1
        self.st_align_lw = 2 / (1 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - 1

    def train_loop(self, epoch, base_loader, optimizer, params=None):
        print_freq = 10
        avg_loss = 0
        avg_ts_loss = 0
        avg_st_loss = 0

        if not isinstance(base_loader, dict):
            train_loader = base_loader
        else:
            train_loader = base_loader['base']

        if (params.st_align or params.ts_align) and epoch == 0:
            unlabeled_loader = self.get_pseudo_loader(base_loader['unlabeled'], soft_label=True, num_img=2)
            base_loader['unlabeled'] = unlabeled_loader

        for i, (x, y) in enumerate(train_loader):
            if params.ts_align or params.st_align:
                try:
                    ux, uy, *_ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(base_loader['unlabeled'])
                    ux, uy, *_ = next(unlabeled_iter)
                ux1, ux2, uy = ux[0].cuda(), ux[1].cuda(), uy.cuda()
                x1, x2, y = x[0].cuda(), x[1].cuda(), y.cuda()
                #print("x1:",x1.shape)
                #print("x2:",x2.shape)
                #print("y:",y)
                # get pseudo label
                with torch.no_grad():
                    fux1 = self.feature(ux1)
                    pred = self.classifier(fux1)
                    pred = F.softmax(pred, dim=-1)
                    #print("pred:",pred)
                    #print("pred.shape:",pred.shape)
                    pred = params.mix_lambda * uy + (1-params.mix_lambda)*pred
                    pseudo_label = pred.max(dim=-1)[1].detach()
                    confidence = pred.max(dim=-1)[0].detach()
                    mask = confidence.ge(self.params.threshold)

                # get features
                pseudo_label0 = pseudo_label
                ux2, pseudo_label = ux2[mask], pseudo_label[mask]
                x_ux = torch.cat([x1, x2, ux2])
                fx_fux = self.feature(x_ux)
                fx1, fx2, fux2 = torch.split(fx_fux, [x1.shape[0], x2.shape[0], ux2.shape[0]])

                # classification head for labeled images.
                loss = self.loss_fn(self.classifier(fx1), y)
                avg_loss = avg_loss + loss.item()

                # ts_align
                if params.ts_align:
                    if pseudo_label.shape[0] > 0:
                        dist = ((F.normalize(fux2).unsqueeze(1) - F.normalize(self.classifier.weight).unsqueeze(0)) ** 2).sum(-1)
                        dist *= self.ts_align_tau
                        ts_align_loss = self.loss_fn(-dist, pseudo_label)
                        ts_align_loss *= (mask.float().sum()) / mask.shape[0]
                    else:
                        ts_align_loss = torch.tensor(0.).cuda()
                    avg_ts_loss += ts_align_loss.item()
                    loss += ts_align_loss

                # st_align
                if params.st_align:
                    dist = ((F.normalize(fx2).unsqueeze(1) - F.normalize(self.target_proto).unsqueeze(0)) ** 2).sum(-1)
                    dist *= self.st_align_tau
                    st_align_loss = self.loss_fn(-dist, y)
                    avg_st_loss += st_align_loss.item()
                    loss += self.st_align_lw * st_align_loss

                    self.update_target_proto(fux1, pseudo_label0)

            else:
                loss, fx = self.forward_loss(x, y)
                avg_loss = avg_loss + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_freq == 0:
                print_line = 'Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1))
                if self.params.st_align:
                    print_line += ' | st_loss {:f}'.format(avg_st_loss / (i + 1))
                if self.params.ts_align:
                    print_line += ' | ts_loss {:f}'.format(avg_ts_loss / (i + 1))
                print(print_line)

        loss_dict = {'loss': avg_loss / (i + 1)}
        if self.params.st_align:
            loss_dict['st_loss'] = avg_st_loss / (i + 1)
        if self.params.ts_align:
            loss_dict['ts_loss'] = avg_ts_loss / (i + 1)
        return loss_dict

    def metatrain_loop(self,epoch, base_loader, optimizer, params):
        avg_metaloss = 0
        #unlabeled_loader = self.get_pseudo_loader(base_loader['unlabeled'], soft_label=True, num_img=2)
        #base_loader['unlabeled'] = unlabeled_loader
        metabase_loader = base_loader['metabase']
        for i, (img, label) in enumerate(metabase_loader):
            img, label = img.squeeze(), label.squeeze()
            #print("img:",img.shape)
            #print("label:",label.shape)
            n_way = params.test_n_way
            n_shot = params.n_shot
            n_query = img.shape[1] - n_shot

            support_label = torch.arange(n_way).unsqueeze(1).repeat(1, n_shot).view(-1)
            query_label = torch.from_numpy(np.repeat(range(n_way ), n_query ))
            #print("slabel:",support_label)
            #print("qlabel:",query_label)
            img.requires_grad_(True)
            img = img.cuda()
            #print("img:",img.requires_grad)
            
            img = img.view(-1, *img.shape[2:])
            features = self.feature(img)
            #print("feature:",features.shape)
            features = F.normalize(features, dim=1)
            features = features.view(n_way, n_shot + n_query, -1)
            support_feature = features[:, :n_shot].reshape(n_way, n_shot, -1)
            query_feature = features[:, n_shot:].reshape(n_way*n_query, -1)
            

            query_xf = query_feature
            support_proto = support_feature.mean(-2) 

            scores = -self.l2distance(query_xf.transpose(-2, -1).contiguous(), support_proto.transpose(-2, -1).contiguous())
            scores= scores.view(75,-1)
            #print("scores111:",scores)
            #query_xf = query_xf.view(n_way* n_query, -1)
            #support_proto = support_proto.view(n_way,-1)
            #dists = -self.euclidean_dist(query_xf, support_proto)
            #print("scores222:",dists)
            #_, predict_labels = torch.max(scores, 1)
            #print("predict_labels:",predict_labels)

            query_label = query_label.cuda()
            #metaloss = nn.CrossEntropyLoss(scores, query_label)

            metaloss = torch.nn.functional.cross_entropy(scores, query_label)

            #logscores = F.log_softmax(scores,dim=1)
            #print("scores:",logscores)
            #print("query_label",query_label)
            #loss = nn.NLLLoss(logscores,query_label.view(1,75))
            #print("loss:",loss)

            #print("metaloss:",metaloss.requires_grad)
            avg_metaloss = avg_metaloss + metaloss.item()

            optimizer.zero_grad()
            metaloss.backward()
            optimizer.step()
            #print(i)
            if i % 10 == 0:
                print_line = 'Epoch {:d} || Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(metabase_loader),metaloss)
                print(print_line)
        loss_dict = {'avg_metaloss': avg_metaloss}
        return loss_dict

    def metatest_loop(self, epoch, val_loader, params):
        accs = []
        for img, label in tqdm(val_loader):
            img, label = img.squeeze(), label.squeeze()
            n_way = params.test_n_way
            n_shot = params.n_shot
            n_query = img.shape[1] - n_shot

            support_label = torch.arange(n_way).unsqueeze(1).repeat(1, n_shot).view(-1)
            query_label = torch.arange(n_way).unsqueeze(1).repeat(1, n_query).view(-1)

            img = img.cuda()
            #print("img:",img.requires_grad)
            
            img = img.view(-1, *img.shape[2:])
            features = self.feature(img)
            #print("feature:",features.shape)
            features = F.normalize(features, dim=1)
            features = features.view(n_way, n_shot + n_query, -1)
            support_feature = features[:, :n_shot].reshape(n_way, n_shot, -1)
            query_feature = features[:, n_shot:].reshape(n_way*n_query, -1)
            

            query_xf = query_feature
            support_proto = support_feature.mean(-2) 

            scores = -self.l2distance(query_xf.transpose(-2, -1).contiguous(), support_proto.transpose(-2, -1).contiguous())
            scores= scores.view(75,-1)
            query_label = query_label.cuda()

            #_, predict_labels = torch.max(scores, 1)
            #print("predict_labels:",predict_labels)
            #loss = nn.CrossEntropyLoss(scores, query_label)
            #metaloss = torch.nn.functional.cross_entropy(scores, query_label)
            logscores = F.softmax(scores,dim=1)
            print("scores:",logscores)
            #print("query_label",query_label)
            #loss = nn.NLLLoss(logscores,query_label.view(1,75))
            #print("loss:",loss)
            #print("metaloss:",metaloss.requires_grad)
            #avg_metaloss = avg_metaloss + metaloss.item()
            #optimizer.zero_grad()
            #metaloss.backward()
            #optimizer.step()
            #print(i)

            _, predict_labels = torch.max(scores, 1)
            print("predict_labels:",predict_labels)
            rewards = [1 if predict_labels[j]==query_label[j].to(predict_labels.device) else 0 for j in range(75)]
            #print("rewards:",np.array(rewards).sum()/75*100)
            accs.append(np.array(rewards).sum()/75*100)
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        print('Epoch %d, Test Acc = %4.2f%% +- %4.2f%%' % (epoch, acc_mean, 1.96 * acc_std / np.sqrt(len(accs))))
        return acc_mean

    def test_loop(self, epoch, val_loader, params):
        accs = []
        with torch.no_grad():
            for img, label in tqdm(val_loader):
                img, label = img.squeeze(), label.squeeze()
                n_way = params.test_n_way
                n_shot = params.n_shot
                n_query = img.shape[1] - n_shot

                support_label = torch.arange(n_way).unsqueeze(1).repeat(1, n_shot).view(-1).numpy()
                query_label = torch.arange(n_way).unsqueeze(1).repeat(1, n_query).view(-1).numpy()

                img = img.cuda()
                img = img.view(-1, *img.shape[2:])
                features = self.feature(img)
                features = F.normalize(features, dim=1)
                features = features.view(n_way, n_shot + n_query, -1)
                support_feature = features[:, :n_shot].detach().cpu().numpy().reshape(n_way * n_shot, -1)
                query_feature = features[:, n_shot:].detach().cpu().numpy().reshape(n_way * n_query, -1)

                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1.0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_feature, support_label)
                scores = clf.predict_proba(query_feature)
                print("scores:",scores)
                query_pred = clf.predict(query_feature)
                #print("querypred:",query_pred)
                #print("querypred.shape:",query_pred.shape)
                acc = np.equal(query_pred, query_label).sum() / query_label.shape[0]
                accs.append(acc * 100)
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        print('Epoch %d, Test Acc = %4.2f%% +- %4.2f%%' % (epoch, acc_mean, 1.96 * acc_std / np.sqrt(len(accs))))
        return acc_mean
