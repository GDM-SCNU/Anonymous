import dgl
import numpy as np
import torch
import scipy.io as scio
import scipy.sparse as sp
def load_data(name: str):
    if name == "imdb5k":

        data = scio.loadmat("datasets/" + name)
        adj_list = []
        label = data['label']
        attr = data['feature']
        num_nodes, feat_dim = attr.shape
        communities = label.shape[1]

        labels = np.argmax(label, axis=1)
        topo_mdm = dgl.from_scipy(sp.csr_matrix(data['MDM']))
        topo_mam = dgl.from_scipy(sp.csr_matrix(data['MAM']))

        adj_list.append(topo_mdm)
        adj_list.append(topo_mam)

        for i, netw in enumerate(adj_list):
            homo = cal_homophily(netw.adjacency_matrix().to_dense().numpy(), labels, num_nodes)
            print(f"{i}-layer, homophily:{homo}")
        attr = torch.from_numpy(attr).to(torch.float32)
        return num_nodes, feat_dim, communities, labels, adj_list, attr
    elif name == "acm":
        data = scio.loadmat("datasets/" + name)
        adj_list = []
        label = data['label']
        attr = data['feature']
        num_nodes, feat_dim = attr.shape
        communities = label.shape[1]

        labels = np.argmax(label, axis=1)

        topo_plp = dgl.from_scipy(sp.csr_matrix(data['PLP']))
        topo_pap = dgl.from_scipy(sp.csr_matrix(data['PAP']))

        adj_list.append(topo_plp)
        adj_list.append(topo_pap)
        attr = torch.from_numpy(attr).to(torch.float32)

        for i, netw in enumerate(adj_list):
            homo = cal_homophily(netw.adjacency_matrix().to_dense().numpy(), labels, num_nodes)
            print(f"{i}-layer, homophily:{homo}")


        return num_nodes, feat_dim, communities, labels, adj_list, attr
    elif name == "dblp":
        data = scio.loadmat("datasets/" + name)
        adj_list = []
        label = data['label']
        attr = data['features']
        num_nodes, feat_dim = attr.shape
        communities = label.shape[1]

        labels = np.argmax(label, axis=1)

        topo_aptpa = dgl.from_scipy(sp.csr_matrix(data['net_APTPA']))
        topo_apcpa = dgl.from_scipy(sp.csr_matrix(data['net_APCPA']))
        topo_apa = dgl.from_scipy(sp.csr_matrix(data['net_APA']))

        adj_list.append(topo_aptpa)
        adj_list.append(topo_apcpa)
        adj_list.append(topo_apa)

        for i, netw in enumerate(adj_list):
            homo = cal_homophily(netw.adjacency_matrix().to_dense().numpy(), labels, num_nodes)
            print(f"{i}-layer, homophily:{homo}")

        attr = torch.from_numpy(attr).to(torch.float32)
        return num_nodes, feat_dim, communities, labels, adj_list, attr
    else:
        print("ERROR")

def cal_homophily(adj, label, num_nodes):
    #datasets = dgl.data.CoraGraphDataset()
    #cora = datasets[0]
    #label = cora.ndata['label'].numpy()
    #num_nodes = cora.num_nodes()
    communities = np.zeros((num_nodes, len(np.unique(label))))
    communities[np.arange(num_nodes), label] = 1
    node_communities = communities.dot(communities.T)
    #adj = cora.adjacency_matrix().to_dense().numpy()
    deg = adj.sum(axis = 1)
    res =  (adj * node_communities).sum(axis=1)
    res = res / deg
    return res.sum() / num_nodes
    # print(res.sum() / num_nodes)

if __name__ == "__main__":
    load_data("dblp")