import torch
import torch.nn as nn
import myKit
import warnings
warnings.filterwarnings("ignore")

""""具体训练参数设置"""

if __name__ == '__main__':

    lr = 5e-4
    batch_size = 32
    # batch_size = 8
    num_epochs = 50
    weight_decay = 0.0001
    lr_period = 10
    lr_decay = 0.5
    M = 4
    alpha = 1
    beta = 1
    gamma = 1
    lambd = 1

    male_net = myKit.get_net(M)
    female_net = myKit.get_net(M)
    # bone_dir = os.path.join('..', 'data', 'archive', 'testDataset')
    bone_dir = "../archive"
    csv_name = "boneage-training-dataset.csv"
    # train_df, valid_df = myKit.split_data(bone_dir, csv_name, 20, 0.1, 512)
    male_train_df, male_valid_df, female_train_df, female_valid_df = myKit.split_data(bone_dir, csv_name, 20, 0.1, 512)
    male_train_set, male_val_set = myKit.create_data_loader(male_train_df, male_valid_df)
    female_train_set, female_val_set = myKit.create_data_loader(female_train_df, female_valid_df)
    torch.set_default_tensor_type('torch.FloatTensor')
    myKit.map_fn(net=male_net, train_dataset=male_train_set, valid_dataset=male_val_set, num_epochs=num_epochs, lr=lr, wd=weight_decay, lr_period=lr_period, lr_decay=lr_decay, alpha=alpha, beta=beta, gamma=gamma, lambd=lambd, batch_size=batch_size, 
                 model_path="model_RA_male.pth", record_path="RECORD_RA_male.csv")
    myKit.map_fn(net=female_net, train_dataset=female_train_set, valid_dataset=female_val_set, num_epochs=num_epochs, lr=lr, wd=weight_decay, lr_period=lr_period, lr_decay=lr_decay, alpha=alpha, beta=beta, gamma=gamma, lambd=lambd, batch_size=batch_size, 
                 model_path="model_RA_female.pth", record_path="RECORD_RA_female.csv")

