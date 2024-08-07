import torch
import torch.nn as nn
import numpy as np
from module.mask_model import *
from module.inv_loss import *
from scatter import scatter
from module.fairness import *
import networkx as nx

def gini_index(p, device):
    n = p.shape[0]
    p, indices = torch.sort(p)
    k = (n+1) - torch.arange(1,n+1).to(device)
    numerator = torch.sum(k*p)*2
    denomitor = n * torch.sum(p)
    return (n+1)/n - (numerator/denomitor)

class MF(nn.Module):
    def __init__(self, args, data):
        super(MF, self).__init__()
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.decay = args.regs
        self.device = torch.device(args.cuda)
        self.saveID = args.saveID

        self.train_user_list = data.train_user_list
        self.valid_user_list = data.valid_user_list
        # = torch.tensor(data.population_list).cuda(self.device)
        self.user_pop = torch.tensor(data.user_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.item_pop = torch.tensor(data.item_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.user_pop_max = data.user_pop_max
        self.item_pop_max = data.item_pop_max

        self.embed_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embed_item = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    # Prediction function used when evaluation
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        users = self.embed_user(torch.tensor(users).cuda(self.device))
        items = torch.transpose(self.embed_item(torch.tensor(items).cuda(self.device)), 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()


class ADV_DROP(MF):
    def __init__(self, args, data, writer):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph().cuda(self.device)
        self.args = args
        if self.args.dropout_type == 1:
            self.edge_index = data.getEdgeIndex().cuda(self.device)
        self.n_layers = args.n_layers
        self.inv_loss = Inv_Loss_Embed(args)
        self.M = Mask_Model_Attention(args)
        self.warmup = True
        self.sigmoid = nn.Sigmoid()

        self.embed_user_dual = nn.Embedding(self.n_users, self.emb_dim)
        self.embed_item_dual = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.xavier_normal_(self.embed_user_dual.weight)
        nn.init.xavier_normal_(self.embed_item_dual.weight)
        self.is_train=True
        self.writer = writer
        self.global_step=0
        self.user_tags=[]
        self.item_tags=[]
        if 'ml' in args.dataset:
            self.user_tags = data.get_user_tags()
        
        if 'coat' in args.dataset:
            self.user_tags = data.get_user_tags()
            self.item_tags = data.get_item_tags()
        
        if self.args.use_attribute:
            self.user_feature_embed = []
            self.item_feature_embed = []
            self.generate_embedings(self.user_tags, self.user_feature_embed)
            self.generate_embedings(self.item_tags, self.item_feature_embed)
            self.user_dense = nn.Linear(self.emb_dim* (len(self.user_tags)+1) ,self.emb_dim)
            self.user_dense_dual = nn.Linear(self.emb_dim*(len(self.user_tags)+1),self.emb_dim)
            self.item_dense = nn.Linear(self.emb_dim*(len(self.item_tags)+1),self.emb_dim)
            self.item_dense_dual = nn.Linear(self.emb_dim*(len(self.item_tags)+1),self.emb_dim)

            
    def generate_embedings(self, tags, feature_embed):
        featuren_len = len(tags) 
        if featuren_len > 0:
            for i in range(featuren_len):
                max_value = torch.max(tags[i])+1
                embed = nn.Embedding(max_value, self.emb_dim).to(self.device)
                nn.init.xavier_normal_(embed.weight)
                feature_embed.append(embed)
                # feature_embed[i] = embed

    def concat_features(self):
        user_features = []
        for i in range(len(self.user_feature_embed)):
            # print(len(self.user_feature_embed), len(self.user_tags))
            # print(self.user_feature_embed[i].weight.shape)
            user_features.append(self.user_feature_embed[i].weight[self.user_tags[i].to(torch.int64)])

        item_features = []
        for i in range(len(self.item_feature_embed)):
            item_features.append(self.item_feature_embed[i].weight[self.item_tags[i].to(torch.int64)])

        if len(user_features)>0:
            user_features = torch.cat(user_features,1)
        if len(item_features)>0:
            item_features = torch.cat(item_features,1)

        return user_features, item_features
    

    def compute_mask_gini(self, mask, index, view='user'):
        # get edge_user_index 
        edge_user_index = torch.where(self.edge_index[0,:] < self.n_users, self.edge_index[0,:], self.edge_index[1,:])
        edge_item_index = torch.where(self.edge_index[0,:] < self.n_users, self.edge_index[1,:]-self.n_users, self.edge_index[0,:]-self.n_users)
        device = self.user_tags[index].device if view == 'user' else self.item_tags[index].device

        if view == 'user':
            edge_user_index = edge_user_index.to(device)
            edge_attribute = self.user_tags[index][edge_user_index].to(torch.int64).to(self.device)
        else:
            edge_item_index = edge_item_index.to(device)
            edge_attribute = self.item_tags[index][edge_item_index].to(torch.int64).to(self.device)
        kk = scatter(mask, edge_attribute, dim=0, reduce="mean")
        return gini_index(kk, self.device), kk
    
    def compute_cluster_loss(self, mask, index, view = 'user'):
        # get edge_user_index 
        edge_user_index = torch.where(self.edge_index[0,:] < self.n_users, self.edge_index[0,:], self.edge_index[1,:])
        edge_item_index = torch.where(self.edge_index[0,:] < self.n_users, self.edge_index[1,:]-self.n_users, self.edge_index[0,:]-self.n_users)
        device = self.user_tags[index].device if view == 'user' else self.item_tags[index].device

        if view == 'user':
            edge_user_index = edge_user_index.to(device)
            edge_attribute = self.user_tags[index][edge_user_index].to(torch.int64).to(self.device)
        else:
            edge_item_index = edge_item_index.to(device)
            edge_attribute = self.item_tags[index][edge_item_index].to(torch.int64).to(self.device)
        kk = scatter(mask, edge_attribute, dim=0, reduce="mean")
        kk = kk.reshape((1,-1))
        loss = torch.mean(torch.pow((kk - kk.T)**2 + 1e-10, 1/2))
        return loss, kk

        
    def draw_graph_init(self, mask, start='user'):
        G = nx.DiGraph()
        edges = self.edge_index.cpu().numpy().T
        new_mask=[]
        for i in range(len(edges)):
            e = edges[i]
            if start=='user':
                if e[0]<self.n_users:
                    G.add_edge(e[0], e[1], weight=mask[i])
                    new_mask.append(mask[i])
            else:
                if e[0]>=self.n_users:
                    G.add_edge(e[0], e[1], weight=mask[i])
                    new_mask.append(mask[i])
        edge_labels = nx.get_edge_attributes(G, "weight")
        return G, edge_labels,new_mask

    def add_node_tag(self, G, user_index, item_index):
        node_attribute_user = self.user_tags[user_index]
        node_attribute_item = self.item_tags[item_index]
        node_attribute_all = torch.cat((node_attribute_user, node_attribute_item),0).numpy()
        for i in range(self.n_users + self.n_items):
            if i< self.n_users:
                G.add_node(i, feature= node_attribute_all[i])
            else:
                G.add_node(i, feature= node_attribute_all[i])
    
        labels = nx.get_node_attributes(G, 'feature')
        return G, labels

    def step(self):
        self.global_step+=1

    def __dropout(self, graph, keep_prob, mask, is_arm=False):
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        if not is_arm:
            if self.args.dropout_type == 0:
                random_index = torch.cuda.FloatTensor(len(values)).uniform_().cuda(self.device) + keep_prob
                # random_index = torch.rand(len(values)).cuda(self.device) + keep_prob
            else:
                random_index = torch.cuda.FloatTensor(len(values)).uniform_().cuda(self.device) + mask
                # random_index = torch.rand(len(values)).cuda(self.device) + mask
            random_index = random_index.int().bool()
        else:
            random_index = mask
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def compute(self, dual=False, dropout = False, mask = None):
        final_embed_user, final_embed_item = None, None
        if self.args.use_attribute:
            combined_user_feature, combined_item_feature = self.concat_features()
            if len(self.user_feature_embed) > 0:
                final_embed_user = torch.cat([combined_user_feature, self.embed_user_dual.weight],1) if dual else torch.cat([combined_user_feature, self.embed_user.weight],1)
            if len(self.item_feature_embed) > 0:
                final_embed_item = torch.cat([combined_item_feature, self.embed_item_dual.weight],1) if dual else torch.cat([combined_item_feature, self.embed_item.weight],1)

        is_arm = True if mask != None else False

        if not dual:
            users_emb = self.user_dense(final_embed_user) if final_embed_user is not None else  self.embed_user.weight
            items_emb = self.item_dense(final_embed_item) if final_embed_item is not None else self.embed_item.weight
        else:
            users_emb = self.user_dense_dual(final_embed_user) if final_embed_user is not None else  self.embed_user_dual.weight
            items_emb = self.item_dense_dual(final_embed_item) if final_embed_item is not None else self.embed_item_dual.weight
        
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        if dropout:
            if mask == None:
                if self.args.dropout_type == 0:
                    mask = None
                else:
                    mask = self.M(all_emb, self.edge_index) if dual else 1 - self.M(all_emb, self.edge_index)
            g_droped = self.__dropout(self.Graph, self.args.keep_prob, mask, is_arm).cuda(self.device)
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items
    
    def regularize(self, users, pos_items, neg_items, dual=False):
        if not dual:
            userEmb0 = self.embed_user(users)
            posEmb0 = self.embed_item(pos_items)
            negEmb0 = self.embed_item(neg_items)
        else:
            userEmb0 = self.embed_user_dual(users)
            posEmb0 = self.embed_item_dual(pos_items)
            negEmb0 = self.embed_item_dual(neg_items)
        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        return regularizer
    

    def forward(self, users, pos_items, neg_items, is_draw=False, is_cluster = False):
        mf_loss=0
        reg_loss=0
        user_embeds=[]
        item_embeds=[]


        for dual_ind in [True,False]:
            all_users, all_items =  self.compute(dual = dual_ind, dropout=True)
            if dual_ind and self.args.dropout_type == 1:
                if is_draw:
                    mask = self.get_mask(dual_ind)
                    self.writer.add_histogram('Dropout Mask', mask, self.global_step)
                    for a_index in range(len(self.user_tags)):
                        #gini_value, kk  = self.compute_mask_gini(mask, a_index,'user')
                        dist_value, kk  = self.compute_cluster_loss(mask, a_index,'user')
                        self.writer.add_scalar(f'Attribute_Dist/User Attribute {a_index}', dist_value, self.global_step)
                        
                        #self.writer.add_scalar(f'Attribute_Gini/User Attribute {a_index}', gini_value, self.global_step)
                        self.writer.add_histogram(f'Attribute_Distribution/User Attribute {a_index}',kk, self.global_step)
                        #self.writer.add_scalars(f'Attribute_Means/User Attribute Distribution {a_index}', {f"group {i}":kk[i] for i in range(len(kk))}, self.global_step)
                    for a_index in range(len(self.item_tags)):
                        #print(self.item_tags[a_index].shape)
                        #gini_value, kk  = self.compute_mask_gini(mask, a_index, 'item')
                        #self.writer.add_scalar(f'Attribute_Gini/Item Attribute {a_index}', gini_value, self.global_step)
                        dist_value, kk  = self.compute_cluster_loss(mask, a_index,'item')
                        self.writer.add_scalar(f'Attribute_Dist/Item Attribute {a_index}', dist_value, self.global_step)
                        self.writer.add_histogram(f'Attribute_Distribution/Item Attribute {a_index}',kk, self.global_step)
                        #self.writer.add_scalars(f'Attribute_Means/Item Attribute Distribution {a_index}', {f"group {i}":kk[i] for i in range(len(kk))}, self.global_step)

            user_embeds.append(all_users)
            item_embeds.append(all_items)

            users_emb = all_users[users]
            pos_emb = all_items[pos_items]
            neg_emb = all_items[neg_items].squeeze()

            pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
            neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

            regularizer = self.regularize(users, pos_items, neg_items, dual_ind) / self.batch_size

            maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

            mf_loss = mf_loss + torch.negative(torch.mean(maxi))
            reg_loss = reg_loss+ self.decay * regularizer
        
        #print(type(self.args.inv_tau))
        
        inv_loss, losses= self.inv_loss(user_embeds, item_embeds)
        inv_loss = self.args.inv_tau*inv_loss
        #inv_loss = -self.inv_loss(item_embeds[0], item_embeds[1], user_embeds[0], user_embeds[1], users)
        if is_cluster:
            mask = self.get_mask(True)
            print("------")
            for i in range(len(self.user_tags)):
                print(self.compute_cluster_loss(mask, i)[0])
            print("------")
        return mf_loss, reg_loss, inv_loss


    def get_mask(self, dual_ind):
        if not dual_ind:
            users_emb = self.embed_user.weight
            items_emb = self.embed_item.weight
        else:
            users_emb = self.embed_user_dual.weight
            items_emb = self.embed_item_dual.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]

        mask = self.M(all_emb, self.edge_index) if dual_ind else 1 - self.M(all_emb, self.edge_index) 

        return mask

    def forward_ARM(self):

        u=torch.rand(len(self.Graph.values())).cuda(self.device)

        mf_loss=0
        reg_loss=0
        user_embeds=[[],[]]
        item_embeds=[[],[]]
        cluster_loss_inv1 = 0 
        cluster_loss_inv2 = 0 
        
        for dual_ind in [True,False]:
            mask = self.get_mask(dual_ind)

            if dual_ind:
                self.writer.add_histogram('Dropout Mask', mask, self.global_step)
            drop1 = u > 1 - mask
            drop2 = u < mask
            if self.args.use_mask_inv:
                for i in range(len(self.user_tags)):
                    if dual_ind:
                        cluster_loss_inv1 += self.compute_cluster_loss(drop1.to(torch.float), i)[0]
                    else: 
                        cluster_loss_inv2 += self.compute_cluster_loss(drop2.to(torch.float), i)[0]
            # print("drop1 shape: ", drop1.shape)
            # print("count", torch.sum(drop1))

            # print("drop2 shape: ", drop2.shape)
            # print("count", torch.sum(drop2))

            for idx, drop in enumerate([drop1, drop2]):
                all_users, all_items =  self.compute(dual = dual_ind, dropout=True, mask=drop)
                user_embeds[idx].append(all_users)
                item_embeds[idx].append(all_items)

        a = self.inv_loss(user_embeds[0], item_embeds[0])[0]
        b = self.inv_loss(user_embeds[1], item_embeds[1])[0]
        if self.args.use_mask_inv:
            inv_loss1 = a + cluster_loss_inv1*self.args.cluster_coe 
            inv_loss2 = b + cluster_loss_inv2*self.args.cluster_coe 
        else:
            inv_loss1 = a 
            inv_loss2 = b 
        # print("inv loss 1", inv_loss1)
        # print(user_embeds[0].shape, item_embeds[0].shape)
        # print("inv loss 2", inv_loss2)
        my_grad = self.args.grad_coeff * (-inv_loss1 + inv_loss2) * (u-0.5) 
        return my_grad
        
    
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()
    
    def freeze_args(self, adv=True):
        if adv:
            self.embed_user.requires_grad_(False)
            self.embed_item.requires_grad_(False)
            self.embed_user_dual.requires_grad_(False)
            self.embed_item_dual.requires_grad_(False)
            if self.args.use_attribute:
                for u in self.user_feature_embed:
                    u.requires_grad_(False)
                for i in self.item_feature_embed:
                    i.requires_grad_(False)

                self.user_dense.requires_grad_(False)
                self.user_dense_dual.requires_grad_(False)
                self.item_dense.requires_grad_(False)
                self.item_dense_dual.requires_grad_(False)
                
            
            for param in self.M.parameters():
                param.requires_grad = True
        else:
            self.embed_user.requires_grad_(True)
            self.embed_item.requires_grad_(True)
            self.embed_user_dual.requires_grad_(True)
            self.embed_item_dual.requires_grad_(True)
            if self.args.use_attribute:
                for u in self.user_feature_embed:
                    u.requires_grad_(True)
                for i in self.item_feature_embed:
                    i.requires_grad_(True)

                self.user_dense.requires_grad_(True)
                self.user_dense_dual.requires_grad_(True)
                self.item_dense.requires_grad_(True)
                self.item_dense_dual.requires_grad_(True)

            for param in self.M.parameters():
                param.requires_grad = False

    def new_predict(self, user_idx, item_idx):
        all_users, all_items = self.compute()
        users_emb = all_users[user_idx]
        all_emb = all_items[item_idx]

        pred = torch.sum(torch.mul(users_emb, all_emb), dim=1)
        pred = self.sigmoid(pred)

        return pred.detach().cpu().numpy()

    def get_top_embeddings(self):
        all_users, all_items = self.compute()
        return all_users
    def get_bottom_embeddings(self):
        return self.embed_user.weight

    def get_predict_bias(self, bs=1024):
        all_users, all_items = self.compute()
        users = list(range(self.n_users))
        start=0
        users = items = torch.transpose(all_users[torch.tensor(users).cuda(self.device)], 0, 1)
        user_attributes=[]
        for index in range(len(self.user_tags)):
            user_attributes.append(self.user_tags[index].to(torch.int64).to(self.device))
        bias_scores=torch.zeros(len(self.user_tags)).to(self.device)
        while start < self.n_items:
            end = start + bs if start + bs < self.n_items else self.n_items
            item_idx=np.arange(start, end)
            start=end
            items = all_items[torch.tensor(item_idx).cuda(self.device)]
            rate_batch = torch.sigmoid(torch.matmul(items, users))
            
            bias_score=[]
            for attr in user_attributes:
                grp_avg=scatter(rate_batch, attr, dim=1, reduce="mean")
                grp_bias = torch.sum(torch.max(grp_avg, dim=1).values - torch.min(grp_avg, dim=1).values)
                # print(grp_bias)
                bias_score.append(grp_bias)
            bias_scores = bias_scores + torch.stack(bias_score)
        bias_scores = bias_scores / self.n_users
        return bias_scores.detach().cpu().numpy()
        


class ADV_DROP_BCE(ADV_DROP):
    def __init__(self, args, data, writer):
        super().__init__(args, data, writer)
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
    def forward(self, users, pos_items, neg_items):
        mf_loss=0
        reg_loss=0
        user_embeds=[]
        item_embeds=[]

        for dual_ind in [True,False]:
            all_users, all_items =  self.compute(dual = dual_ind, dropout=True)
            if dual_ind and self.args.dropout_type == 1:
                mask = self.get_mask(dual_ind)
                self.writer.add_histogram('Dropout Mask', mask, self.global_step)
            user_embeds.append(all_users)
            item_embeds.append(all_items)

            users_emb = all_users[users]
            pos_emb = all_items[pos_items]
            neg_emb = all_items[neg_items]
            all_emb = torch.cat((pos_emb, neg_emb),0)

            userEmb0 = self.embed_user(users)
            posEmb0 = self.embed_item(pos_items)
            negEmb0 = self.embed_item(neg_items)

            pos_label = torch.ones((len(pos_emb),))
            neg_label = torch.zeros((len(neg_emb),))
            all_label = torch.cat((pos_label, neg_label),0)

            pred = torch.sum(torch.mul(torch.cat((users_emb, users_emb),0), all_emb), dim=1)
            pred = self.sigmoid(pred)
            bce_loss = self.bce(pred,all_label.cuda(self.device))

            regularizer = self.regularize(users, pos_items, neg_items, dual_ind) / self.batch_size

            mf_loss = mf_loss + bce_loss
            reg_loss = reg_loss+ self.decay * regularizer
        
        inv_loss, losses=self.args.inv_tau*self.inv_loss(user_embeds, item_embeds)
        #inv_loss = -self.inv_loss(item_embeds[0], item_embeds[1], user_embeds[0], user_embeds[1], users)

        return mf_loss, reg_loss, inv_loss