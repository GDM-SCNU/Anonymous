import numpy as np
import networkx as nx
import scipy.sparse as sp
import os
import evaluate
import torch
import random
import dgl
import pickle as pkl
import scipy.io as scio
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR
import dgl.function as fn
from dgl.nn import GraphConv

random.seed(826)
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)

DATASET = "imdb5k"

def knn_graph(feat, topk, weight = False, loop = True):
    sim_feat = cosine_similarity(feat)
    sim_matrix = np.zeros(shape=(feat.shape[0], feat.shape[0]))

    inds = []
    for i in range(sim_feat.shape[0]):
        ind = np.argpartition(sim_feat[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    for i, vs in enumerate(inds):
        for v in vs:
            if v == i:
                pass
            else:
                if weight is True:
                    sim_matrix[i][v] = sim_feat[i][v]
                    sim_matrix[v][i] = sim_feat[v][i]
                else:
                    sim_matrix[i][v] = 1
                    sim_matrix[v][i] = 1

    sp_matrix = sp.csr_matrix(sim_matrix)
    dgl_matrix = dgl.from_scipy(sp_matrix)
    if loop is True:
        dgl_matrix = dgl.add_self_loop(dgl_matrix)
    return dgl_matrix

def load_data(name: str):
    if name == "imdb5k":

        data = scio.loadmat("datasets/" + "imdb5k")
        adj_list = []
        label = data['label']
        attr = data['feature']
        num_nodes, feat_dim = attr.shape
        communities = label.shape[1]

        labels = np.argmax(label, axis=1)
        topo_mdm = dgl.from_scipy(sp.csr_matrix(data['MDM']))
        topo_mam = dgl.from_scipy(sp.csr_matrix(data['MAM']))
        graph = knn_graph(attr, 18, False) # 6 pretrian 10 train

        adj_list.append(topo_mdm)
        adj_list.append(topo_mam)
        adj_list.append(graph)
        attr = torch.from_numpy(attr).to(torch.float32)
        return num_nodes, feat_dim, communities, labels, adj_list, attr
    else:
        print("ERROR")

class AutoLayer(nn.Module):
    def __init__(self, hid_dim, alpha, eps):
        super(AutoLayer, self).__init__()
        # self.g = graph
        self.gate = nn.Linear(2 * hid_dim, 1)
        self.alpha = alpha
        self.eps = eps

    def edge_applying(self, edges):
        h = torch.cat([edges.dst['h'], edges.src['h']], dim = 1)
        g = torch.tanh(self.gate(h)).squeeze()
        e = g * (edges.dst['d'] * edges.src['d'] * self.alpha + (self.eps - self.alpha) * edges.dst['b'] * edges.src['b'])
        return {'e': e, 'm': g}
    def forward(self, adj, h):
        self.g = adj
        deg = self.g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        self.g.ndata['d'] = norm
        b = torch.zeros((self.g.num_nodes(), 1)).squeeze(-1)
        self.g.ndata['b'] = b

        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', "_"), fn.sum("_", "z"))
        return self.g.ndata['z']

class FLPASS(nn.Module):
    def  __init__(self, hid_dim, out_dim, eps, alpha):
        super(FLPASS, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.t2 = nn.Linear(hid_dim, out_dim, bias = False)
        self.layer = AutoLayer(hid_dim, alpha, eps)
    def forward(self, adj, h):

        h_1 = self.layer(adj, h)
        h = h_1 + self.alpha * h # h_0
        h = self.t2(h)
        return h

class KMVGAE(nn.Module):
    def __init__(self, **kwargs):
        super(KMVGAE, self).__init__()

        self.adj = kwargs['common_adj']
        self.feat = kwargs['feat']
        self.label = kwargs['label']
        self.norm = kwargs['common_norm']
        self.weight_tensor_orig = kwargs['common_weight_tensor_orig']


        self.num_node, self.feat_dim = self.feat.shape
        self.nClusters = len(np.unique(self.label))

        self.view_adj = kwargs['adj_list']
        self.view_norm = kwargs['norm_list']
        self.view_weight = kwargs['weight_list']

        hid1_dim = 32
        self.hid2_dim = 16

        # VGAE training parameters
        self.activation = F.relu

        self.base_gcn = nn.ModuleList([GraphConv(self.feat_dim, hid1_dim, activation = self.activation, bias = False) for i in range(len(self.view_adj))])
        self.gcn_mean = nn.ModuleList([GraphConv(hid1_dim, self.hid2_dim, bias = False) for i in range(len(self.view_adj))])
        # self.gcn_logstddev = nn.ModuleList([GraphConv(hid1_dim, self.hid2_dim) for i in range(len(self.view_adj))])
        # 0.1 0.9
        self.gcn_logstddev = nn.ModuleList(
            [FLPASS(hid1_dim, self.hid2_dim, 0.1, 0.9) for i in range(len(self.view_adj))])


        self.layers = len(self.view_adj)

        self.pi = nn.Parameter(torch.ones(self.nClusters) / self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.hid2_dim), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.hid2_dim), requires_grad=True)


        self.feat_parameter = nn.Parameter(torch.FloatTensor(self.feat_dim, self.hid2_dim))
        torch.nn.init.xavier_uniform_(self.feat_parameter)

        self.gmm = GaussianMixture(n_components=self.nClusters, covariance_type='diag')


    def _encode(self, adj, i):
        hidden = self.base_gcn[i](adj, self.feat)
        self.mean = self.gcn_mean[i](adj, hidden)
        self.logstd = self.gcn_logstddev[i](adj, hidden)
        gaussian_noise = torch.randn(self.num_node, self.hid2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return self.mean, self.logstd, sampled_z

    @staticmethod
    def _decode(z):
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        return A_pred


    def _pretrain(self):
        if not os.path.exists('datasets/'+ DATASET + '/pretrain/dgl_model.pk'):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.008)
            epoch_bar = tqdm(range(100))
            nmi_best = 0
            nmi_list = []
            for _ in epoch_bar:
                optimizer.zero_grad()
                _, _, z = self._encode(self.adj, 1)
                A_pred = self._decode(z)
                loss = self.norm * F.binary_cross_entropy(A_pred.view(-1), self.adj.adjacency_matrix().to_dense().view(-1), weight = self.weight_tensor_orig)
                loss.backward()
                optimizer.step()

                y_pred = self.gmm.fit_predict(z.detach().numpy())
                self.pi.data = torch.from_numpy(self.gmm.weights_).to(torch.float)
                self.mu_c.data = torch.from_numpy(self.gmm.means_).to(torch.float)
                self.log_sigma2_c.data = torch.log(torch.from_numpy(self.gmm.covariances_)).to(torch.float)


                acc = evaluate.cal_acc(self.label, y_pred)
                nmi = evaluate.compute_nmi(y_pred, self.label)
                f1 = evaluate.computer_macrof1(y_pred, self.label)
                ari = evaluate.computer_ari(y_pred, self.label)
                epoch_bar.write('Loss pretraining = {:.4f}, acc = {:.4f}, nmi = {:.4f}, f1 = {:.4f}, ari = {:.4f}'.format(loss, acc , nmi, f1, ari))
                nmi_list.append(acc)
                if (nmi > nmi_best):
                  nmi_best = nmi
                  self.logstd = self.mean

                  torch.save(self.state_dict(), 'datasets/'+ DATASET + '/pretrain/dgl_model.pk')
            print("Best accuracy : ",nmi_best)
        else:
            self.load_state_dict(torch.load('datasets/'+ DATASET + '/pretrain/dgl_model.pk'))

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def _train(self):
        self.load_state_dict(torch.load('datasets/'+ DATASET + '/pretrain/dgl_model.pk'))

        base_gcn_state = self.base_gcn[1].state_dict()
        gcn_mean_state = self.gcn_mean[1].state_dict()
        gcn_logstd_state = self.gcn_logstddev[1].state_dict()

        for i in range(0, len(self.gcn_mean)):
            if i == 1:
                continue
            self.base_gcn[i].load_state_dict(base_gcn_state)
            self.gcn_mean[i].load_state_dict(gcn_mean_state)
            self.gcn_logstddev[i].load_state_dict(gcn_logstd_state)

        optimizer = torch.optim.Adam(self.parameters(),lr=0.002, weight_decay=0.10) # 0.002
        # lr_s = StepLR(optimizer, step_size=10, gamma=0.9)
        acc_list = []
        nmi_list = []
        max_nmi = 0
        epoch_bar = tqdm(range(200)) # 200 for cora & citeseer & pubmed # 100 for blogcatalog
        cm = evaluate.clustering_metrics(self.label)
        e = torch.eye(self.num_node) 
        for epoch in epoch_bar:
            optimizer.zero_grad()
            loss = 0
            z_list = []
            for i in range(len(self.view_adj)):

                z_mean, z_logstd, z = self._encode(self.view_adj[i], i)
                A_pred = self._decode(z)

                loss += self._ELBO_No_W(A_pred, z, z_mean, z_logstd, i)

                tmp_z = torch.softmax(z, dim=1)
                loss_comp = -0.5 * torch.log(torch.det(e - 0.0006 * tmp_z @ tmp_z.T)) # 0.0006
                loss = loss + loss_comp
                z_list.append(z)


            z = torch.sum(torch.stack(z_list, dim=1), dim=1) / len(z_list)


            y_pred, prob_y_pred = self._predict(z)
            cm._set_pred_label(y_pred)
            acc, nmi, f1, ari = cm.evaluationClusterModelFromLabel()

            epoch_bar.write(
                'Loss training = {:.4f}, acc = {:.4f}, nmi = {:.4f}, f1 = {:.4f}, ari = {:.4f}'.format(loss, acc, nmi, f1, ari))
            acc_list.append(acc)
            nmi_list.append(nmi)

            # lr_s.step()
            loss.backward()
            optimizer.step()
        print("MAX ACC : {}, MAX NMI :{}".format(max(acc_list), max(nmi_list)))
    def _ELBO_No_W(self, A_pred, z, z_mean, z_logstd, i):
        pi = self.pi
        mean_c = self.mu_c
        logstd_c = self.log_sigma2_c
        det = 1e-2

        loss_recons = 1e-2 * self.view_norm[i] * F.binary_cross_entropy(A_pred.view(-1), self.view_adj[i].adjacency_matrix().to_dense().view(-1), weight= self.view_weight[i])
        loss_recons = loss_recons * self.num_node

        gamma_c = torch.exp(torch.log(pi.unsqueeze(0)) + self._gaussian_pdfs_log(z, mean_c, logstd_c)) + det
        gamma_c = gamma_c / (gamma_c.sum(1).view(-1,1))

        KL1 = 0.5 * torch.mean(torch.sum(gamma_c * torch.sum(logstd_c.unsqueeze(0) + torch.exp(z_logstd.unsqueeze(1) - logstd_c.unsqueeze(0))
                                                             + (z_mean.unsqueeze(1) - mean_c.unsqueeze(0)).pow(2) / torch.exp(logstd_c.unsqueeze(0)), 2), 1))
        KL2 = torch.mean(torch.sum(gamma_c * torch.log(pi.unsqueeze(0) / (gamma_c)),1)) + 0.5 * torch.mean(torch.sum(1 + z_logstd, 1))
        loss_clus = KL1 - KL2

        loss_elbo = loss_recons + loss_clus
        return loss_elbo

    def _gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self._gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    def _gaussian_pdf_log(self,x,mu,log_sigma2):
        c = -0.5 * torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1)
        return c

    def _predict(self, z):
        pi = self.pi
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        det = 1e-2
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self._gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita = yita_c.detach().numpy()
        return np.argmax(yita, axis=1), yita_c.detach()

def variational_parameter(adj, dgl_adj):
    adj[np.arange(adj.shape[0]), np.arange(adj.shape[0])] = 0
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

    new_dgl_adj = dgl_adj.adjacency_matrix().to_dense()
    new_dgl_adj[np.arange(adj.shape[0]), np.arange(adj.shape[0])] = 1
    weight_mask_orig = new_dgl_adj.view(-1) == 1
    weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
    weight_tensor_orig[weight_mask_orig] = pos_weight_orig

    new_dgl_adj = dgl.from_scipy(sp.csr_matrix(new_dgl_adj))
    return norm, weight_tensor_orig, new_dgl_adj

if __name__ == "__main__":

    num_nodes, feat_dim, k, labels, adj_list, feat = load_data(DATASET)
    common_graph = torch.zeros(size=(num_nodes, num_nodes))

    norm_list = []
    weight_list = []
    adj_recons_list = []

    for i in range(len(adj_list)):
        common_graph += adj_list[i].adjacency_matrix().to_dense()
        norm, weight_tensor_orig, dgl_adj= variational_parameter(adj_list[i].adjacency_matrix().to_dense(), adj_list[i])
        norm_list.append(norm)
        weight_list.append(weight_tensor_orig)
        adj_list[i] = dgl_adj


    common_graph = torch.where(common_graph >=len(adj_list)-1, 1, 0)
    sp_common_graph = sp.csr_matrix(common_graph)
    norm, weight_tensor_orig, new_common_graph = variational_parameter(sp_common_graph, dgl.from_scipy(sp_common_graph))

    configs = {
        "common_adj": new_common_graph,
        "feat": feat,
        "label": labels,
        "common_norm": norm,
        "common_weight_tensor_orig" : weight_tensor_orig,
        "adj_list" : adj_list,
        "norm_list" : norm_list,
        "weight_list": weight_list
    }
    model = KMVGAE(**configs)
    # model._pretrain()
    model._train()
