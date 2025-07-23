# 原版代码
import time
from copy import deepcopy
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer
from dhg.nn import HGNNConv

from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

from HyperMixup import generate_node_pairs, create_e_list_from_incidence_matrix, mixup_criterion


# 给超图hg加节点，给X特征矩阵加节点，给标签张量加节点
def modify_incidence_matrix(hg: Hypergraph, x: torch.Tensor, lbl: torch.Tensor, p, q, l):

    # # 获取需要修改的超边索引和包含它们的节点对及其超边索引--同质性
    # # modify_hyper_index, max_pair_with_e_idx = hg.top20_percent_hyperedges_index()
    # # modify_hyper_index.sort()
    #
    #
    # # 1. 保留 max_pair_with_e_idx 中包含 modify_hyper_index 超边索引的节点对，删除其他节点对
    # filtered_pairs = []
    # # for pair, e_idx_list in max_pair_with_e_idx:
    # #     # 保留包含 modify_hyper_index 超边索引的节点对
    # #
    # #     if e_idx_list[0] in modify_hyper_index:
    # #         filtered_pairs.append((pair, e_idx_list))
    # for (u, v), e_idx_list in max_pair_with_e_idx:
    #     # 这里去除 e_idx_list 部分，直接保留节点对 (u, v)
    #     if any(e_idx in modify_hyper_index for e_idx in e_idx_list):
    #         filtered_pairs.append((u, v))  # 保留符合条件的节点对


    # 将超图的关联矩阵转换为稠密矩阵（dense）
    H_den = hg.H.to_dense().to(device=hg.device)
    # ①训练集节点和节点的超边特征均值-只使用训练集两两节点对不可行
    train_indices, node_pairs = generate_node_pairs(train_mask)
    edge_features_mean = hg.calculate_Xe_mean_of_edge(train_indices, X)

    #②训练集两两节点对和同质性高的节点对求交集节点对
    # filtered_pairs_set = set(filtered_pairs)  # 转换为集合
    # node_pairs_set = set(node_pairs)  # 转换为集合
    # common_pairs = filtered_pairs_set.intersection(node_pairs_set)  # 获取交集
    #num_common_pairs = len(common_pairs)  # 共有节点对的数量

    #③使用训练集两两节点对相似度前n%的节点对
    similar_node_pairs = hg.calculate_top20_uv_similarity(node_pairs, X, l)

    # 新节点的超边关系将被添加到 H_den 的最后
    new_rows = []  # 存储每个新节点的超边关系

    # 动态扩展的新节点特征和标签列表
    new_node_features_list = []  # 存储新节点特征
    new_node_labels_list_a = []    # 存储新节点标签
    new_node_labels_list_b = []  # 存储新节点标签


    # 4. 为每个节点对生成新节点和其对应的超边关系
    for u, v in similar_node_pairs:
        # 计算 u 和 v 的共享超边
        shared_edges = hg.index_edges_between(u, v)  # 假设有这个方法来获取共享的超边
        new_row = torch.zeros(H_den.size(1), dtype=torch.float32)  # 每个新节点的超边关系存储在 new_row 中

        for e_idx in shared_edges:
            new_row[e_idx] = 1.0  # 新节点与共享超边的关系设置为1

        # 其次，对于父节点各自非共同连接的超边，使用常数 p 来选择连接多少超边
        u_edges = hg.N_e(u)
        v_edges = hg.N_e(v)

        # 计算 u 和 v 的非共享超边
        u_non_shared = list(set(u_edges) - set(shared_edges))
        v_non_shared = list(set(v_edges) - set(shared_edges))

        # 随机选择 p 的比例来连接这些非共享超边
        u_selected_edges = random.sample(u_non_shared, k=int(len(u_non_shared) * p)) if u_non_shared else []
        v_selected_edges = random.sample(v_non_shared, k=int(len(v_non_shared) * (1 - p))) if v_non_shared else []

        # 将选择的超边与新节点建立关系
        for e_idx in u_selected_edges + v_selected_edges:
            new_row[e_idx] = 1.0  # 新节点与这些超边的关系设置为1

        # 将该新节点的超边关系添加到 new_rows 中
        new_rows.append(new_row)

        # 5. 计算新节点的特征
        u_features = x[u, :]  # u 节点的特征
        v_features = x[v, :]  # v 节点的特征

        # 获取 u 和 v 对应的超边特征均值
        u_edge_mean = edge_features_mean.get(u, torch.zeros_like(x[0]))  # 默认为零向量
        v_edge_mean = edge_features_mean.get(v, torch.zeros_like(x[0]))  # 默认为零向量

        # 对于每个特征，按 p 和 (1-p) 计算加权和
        new_node_features = p * (q * u_features + (1 - q) * u_edge_mean) + (1 - p) * (q * v_features + (1 - q) * v_edge_mean)
        new_node_features_list.append(new_node_features)





        # 6. 计算新节点的标签

        #①将标签转化为概率限量
        # new_lbl = p * lbl[u] + (1 - p) * lbl[v]
        # new_node_labels_list.append(new_lbl)

        #②以p的大小界定新结点的标签
        # if p > 0.5:
        #     new_node_labels_list.append(lbl[u])  # 选择 u 节点的标签
        # else:
        #     new_node_labels_list.append(lbl[v])  # 选择 v 节点的标签

        #③从损失函数更改分为y_a,y_b
        new_node_labels_list_a.append(lbl[u])
        new_node_labels_list_b.append(lbl[v])


    # 7. 将所有的新节点的超边关系（new_rows）添加到 H_den 的最后
    H_den = torch.cat([H_den, torch.stack(new_rows).to(device=H_den.device)], dim=0)  # 新节点的超边关系被添加为多行

    # 8. 将新节点的特征添加到 x 的最后一行
    new_node_features_tensor = torch.stack(new_node_features_list)  # 转换为张量
    x = torch.cat([x, new_node_features_tensor], dim=0)  # 将新节点的特征添加到 x 的最后一行

    # 9. 将新节点的标签添加到 lbl 的最后一行
    new_node_labels_tensor = torch.tensor(new_node_labels_list_a, dtype=torch.long).to(device=lbl.device) # 转换为张量
    lbl_a = torch.cat([lbl, new_node_labels_tensor], dim=0)  # 将新节点的标签添加到 lbl 的最后一行
    new_node_labels_tensor = torch.tensor(new_node_labels_list_b, dtype=torch.long).to(device=lbl.device)  # 转换为张量
    lbl_b = torch.cat([lbl, new_node_labels_tensor], dim=0)  # 将新节点的标签添加到 lbl 的最后一行

    # 10. 计算添加节点的数量
    num_insert_indices = len(similar_node_pairs)

    # 返回修改后的关联矩阵 H_den、特征矩阵 x 和标签 lbl 以及添加的节点数量
    return H_den, x, lbl_a, lbl_b,  num_insert_indices


class HGNN_hid_mix(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )
        self.hidden_layers = []  # 用于存储隐层表示
        self.module_list = []  # 初始化需要Mix操作的隐层列表
        self.hidden_x_obtaining = False
        self.pre_train = True
        self.now_train = False
        # for n, m in self.named_modules():  # 遍历模型的所有命名模块
        #     if n[:-1] == 'layer':  # 如果模块名以'layer'开头（假设这是需要操作的层）
        #             self.module_list.append(m)  # 将该模块添加到module_list中

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:

        #self.hidden_layers.clear()  # 清空之前的隐层表示
        if  self.pre_train :
            for layer in self.layers:
                X = layer(X, hg)
            return X
        else:
            for i, layer in enumerate(self.layers):
                if i < len(self.layers) - 1:  # 只保存非最后一层的隐层表示
                    if  self.eval() and not self.hidden_x_obtaining:
                        X = layer(X, hg)

                        self.hidden_x_obtaining = True
                        H_pad, X, lbl_a, lbl_b, num_insert_indices = modify_incidence_matrix(hg, X, lbl, lam_p, lam_q, lam_l)
                        hg.num_v = hg.num_v + num_insert_indices
                        e_list = create_e_list_from_incidence_matrix(H_pad.cpu().numpy())
                        hg = Hypergraph(hg.num_v, e_list, device=device)
                        print("\n生成新节点后H的shape")  # 打印训练完成信息
                        print(H_pad.shape)
                        print(X.shape)
                        print(lbl_a.shape)
                        print("\n添加了%d个节点" % num_insert_indices)  # 打印训练完成信息
                        self.hidden_layers.append(X.detach().clone())
                        self.hidden_layers.append(hg.clone())
                        self.hidden_layers.append(lbl_a.detach().clone())
                        self.hidden_layers.append(lbl_b.detach().clone())
                        self.hidden_layers.append(num_insert_indices)
                        print(self.hidden_layers)


                else :
                    X = layer(X, hg)

            return X


def train(net, X, G, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    if not net.hidden_x_obtaining:
        outs = net(X, G)
        outs, lbls = outs[train_idx], lbls[train_idx]
        loss = F.cross_entropy(outs, lbls)

    else:
        outs = net(X, G)

        outs, lbls_a, lbls_b = outs[train_idx], lbls[train_idx], lbl_b[train_idx]  # 获取训练集的输出和标签
        loss = mixup_criterion(outs, lbls_a, lbls_b, lam_p)

    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()







@torch.no_grad()
def infer(net, X, G, lbls, idx, test=False):
    net.eval()
    outs = net(X, G)

    if net.hidden_x_obtaining and not net.now_train:
        idx = net.hidden_layers[2]


    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res


if __name__ == "__main__":


    set_seed(2022)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    data = Cora()
    #data = Pubmed()
    #data = Citeseer()
    X, lbl = data["features"], data["labels"]
    G = Graph(data["num_vertices"], data["edge_list"])
    HG = Hypergraph.from_graph_kHop(G, k=1)
    # HG.add_hyperedges_from_graph_kHop(G, k=1)
    # HG = Hypergraph.from_graph_kHop(G, k=1)
    # HG.add_hyperedges_from_graph_kHop(G, k=2, only_kHop=True)
    print("\n初始H的shape")  # 打印训练完成信息
    print(HG.H.to_dense().shape)




    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    # # 设置低标签比例 rate，例如 0.01
    # rate = 0.015
    # num_total_nodes = train_mask.size(0)
    # num_train_nodes = train_mask.sum().item()  # 140
    # num_select = int(num_total_nodes * rate)
    #
    # # 从 train_mask 为 True 的索引中随机选取 num_select 个
    # true_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
    # selected_indices = random.sample(true_indices, num_select)
    #
    # # 创建新的训练掩码
    # train_mask = torch.zeros_like(train_mask)
    # train_mask[selected_indices] = True
    #
    # # 检查结果
    # print(f"原训练节点数: {num_train_nodes}")
    # print(f"选中的训练节点数: {train_mask.sum().item()}")

    lam_p = 0.49
    lam_q = 0.72
    lam_l = 0.0193


    net = HGNN_hid_mix(data["dim_features"], 16, data["num_classes"])
    # net = HNHN(data["dim_features"], 16, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    HG = HG.to(X.device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, HG, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, HG, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")

                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, HG, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)

    net.pre_train = False
    print("\n预训练完成，网络参数已加载,接下来提取隐层表示")
    res = infer(net, X, HG, lbl, test_mask, test=True)
    net.now_train = True





    new_true_elements = torch.ones(net.hidden_layers[4], dtype=torch.bool)
    new_false_elements = torch.zeros(net.hidden_layers[4], dtype=torch.bool)

    # 使用 torch.cat 将最后一个元素添加到张量尾部
    train_mask = torch.cat((train_mask, new_true_elements), dim=0)
    val_mask = torch.cat((val_mask, new_false_elements), dim=0)
    test_mask = torch.cat((test_mask,new_false_elements), dim=0)

    X = net.hidden_layers[0]
    HG = net.hidden_layers[1]
    lbl_a = net.hidden_layers[2]
    lbl_b = net.hidden_layers[3]


    # Adam优化器同时优化模型的参数和p, q
    optimizer = optim.Adam(
        list(net.parameters()) ,  # 将模型参数和 p, q 一起加入优化器
        lr=0.01,
        weight_decay=5e-4
    )

    X, lbl_a, lbl_b = X.to(device), lbl_a.to(device), lbl_b.to(device)  # 将数据移动到设备
    HG = HG.to(X.device)  # 将超图移动到设备
    net = net.to(device)  # 将模型移动到设备

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(100):
        # train
        train(net, X, HG, lbl_a, train_mask, optimizer, epoch)

        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, HG, lbl_a, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)

    # embeddings = T_sne.get_embeddings(net, X, HG)#{2708，7}
    # T_sne.plot_tsne(embeddings,"HMMixup" , lbl_a.cpu().numpy())

    res = infer(net, X, HG, lbl_a, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)





