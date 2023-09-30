import torch
import torch.nn as nn
import myKit
import warnings
warnings.filterwarnings("ignore")

""""具体训练参数设置"""

if __name__ == '__main__':

    lr = 1e-3
    batch_size = 32
    # batch_size = 8
    num_epochs = 300
    weight_decay = 0
    lr_period = 50
    lr_decay = 0.1
    M = 4
    alpha = 1
    beta = 1
    gamma = 1
    lambd = 1

    net = myKit.get_net(M)
    # bone_dir = os.path.join('..', 'data', 'archive', 'testDataset')
    bone_dir = "../archive"
    csv_name = "boneage-training-dataset.csv"
    train_df, valid_df = myKit.split_data(bone_dir, csv_name, 20, 0.1, 512)
    train_set, val_set = myKit.create_data_loader(train_df, valid_df)
    torch.set_default_tensor_type('torch.FloatTensor')
    myKit.map_fn(net=net, train_dataset=train_set, valid_dataset=val_set, num_epochs=num_epochs, lr=lr, wd=weight_decay, lr_period=lr_period, lr_decay=lr_decay, alpha=alpha, beta=beta, gamma=gamma, lambd=lambd, batch_size=batch_size, 
                 model_path="model_RA.pth", record_path="RECORD_RA.csv")
