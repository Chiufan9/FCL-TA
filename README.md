# FCL‑TA: Optimization Trajectory Alignment for Federated Continual Learning to Mitigate Spatial‑Temporal Trajectory Misalignment
The implementation of FCL‑TA.

## Requirements
The needed libraries are in requirements.txt.

## Dataset preparation:
EMNIST-Letters and CIFAR100 can be automatically downloaded with ```torchvision.datasets```

TinyImageNet can be downloaded from:(http://cs231n.stanford.edu/tiny-imagenet-200.zip). 
After downloading, extract the file to the folder as follows: "datasets/PreciseFCL/TinyImageNet".

HAM10k can be downloaded from: (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
After downloading, extract the file to the folder as follows: "datasets/PreciseFCL/HAM10k".

PBC can be downloaded from: (https://data.mendeley.com/datasets/snkd93bnjr/1).
After downloading, extract the file to the folder as follows: "datasets/PreciseFCL/PBC".

OrganAMNIST (from MedMNIST) can be downloaded from: (https://medmnist.com).
After downloading, extract the file to the folder as follows: "datasets/PreciseFCL/OrganAMNIST".

## Data partition:
To create your own data partitioning file:
        python split_dataset.py 

The generated .pkl file should be saved in the folder "datasets/PreciseFCL".

## Experiments
To run on EMNIST-Letters, excute:

      CUDA_VISIBLE_DEVICES=7 python main.py --dataset EMNIST-Letters --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2024.pkl --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-4 --k_loss_flow 0.5 --k_flow_lastflow 0.4 --flow_explore_theta 0 --device cuda --using_LTC 1 --using_GTR 1 --seed 2024

To run on CIFAR100, excute:

      CUDA_VISIBLE_DEVICES=7 python main.py --dataset CIFAR100 --data_split_file data_split/CIFAR100_split_cn10_tn4_cet20_s2024.pkl --num_glob_iters 40 --local_epochs 400 --lr 1e-3 --flow_lr 5e-3 --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.1 --fedprox_k 0.001 --device cuda --using_LTC 1 --using_GTR 1 --seed 2024

To run on TinyImageNet, excute:

      CUDA_VISIBLE_DEVICES=7 python main.py --dataset TinyImageNet --data_split_file TinyImageNet_cn3_tn6_cet30_cs2_2024.pkl --num_glob_iters 60 --local_epochs 300 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.1 --fedprox_k 0.001 --device cuda --using_LTC 1 --using_GTR 1 --seed 2024

To run on HAM10k, excute:

      CUDA_VISIBLE_DEVICES=7 python main.py --dataset HAM10k_client --data_split_file HAM10k_client_cn3_tn3_cet2_cs1_2001.pkl --num_glob_iters 30 --local_epochs 100 --lr 1e-3 --flow_lr 5e-4 --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.1 --fedprox_k 0.001 --device cuda --using_LTC 1 --using_GTR 1 --seed 2024

To run on PBC, excute:

      CUDA_VISIBLE_DEVICES=7 python main.py --dataset PBC --data_split_file PBC_cn3_tn3_cet2_cs1_2024.pkl --num_glob_iters 30 --local_epochs 150 --lr 1e-3 --flow_lr 5e-4 --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.1 --fedprox_k 0.001 --device cuda --using_LTC 1 --using_GTR 1 --seed 2024

To run on OrganAMNIST, excute:
      
      CUDA_VISIBLE_DEVICES=7 python main.py --dataset MedMNIST --data_split_file OrganAMNIST_cn5_tn4_cet2_cs2_2024.pkl --num_glob_iters 40 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.5 --k_flow_lastflow 0.4 --flow_explore_theta 0 --device cuda --using_LTC 1 --using_GTR 1 --seed 2024

    
## Reference
The code structure is based on the code in [AF-FCL](https://github.com/zaocan666/AF-FCL)
