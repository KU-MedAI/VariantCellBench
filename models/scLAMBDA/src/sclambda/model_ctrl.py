import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import copy

from sclambda.networks import *
from sclambda.utils import *

class Model(object):
    def __init__(self, 
                 #adata, # anndata object already splitted
                 adata_train,
                 adata_val,
                 adata_test,
                 gene_emb, # dictionary for gene embeddings
                 #split_name = 'split',
                 latent_dim = 30, hidden_dim = 512,
                 training_epochs = 20,
                 batch_size = 16,
                 lambda_MI = 200,
                 eps = 0.001,
                 seed = 42,
                 model_path = "models",
                 multi_gene = False,    # 단일변이만 고려 중
                 wandb_run = None,
                 emb_type = 'ALT'
                 ):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        #self.adata = adata.copy()
        self.adata_train = adata_train.copy()
        self.adata_val = adata_val.copy()
        self.adata_test = adata_test.copy()

        self.gene_emb = gene_emb
        self.x_dim = adata_train.shape[1] # gene 개수
        self.p_dim = gene_emb[list(gene_emb.keys())[0]].shape[0]

        self.gene_emb.update({'ctrl': np.zeros(self.p_dim)})
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.lambda_MI = lambda_MI
        self.eps = eps
        self.model_path = model_path
        self.multi_gene = multi_gene
        self.wandb_run = wandb_run
        self.emb_type = emb_type

        # compute perturbation embeddings
        print("Computing %s-dimentisonal perturbation embeddings for %s cells..." % (self.p_dim, adata.shape[0]))
        #self.pert_emb_cells = np.zeros((self.adata.shape[0], self.p_dim))
        self.pert_emb = {}
        # 각 데이터셋에 대해 임베딩 계산 및 .obsm에 저장
        self.adata_train.obsm['pert_emb'] = self._compute_perturbation_embeddings(self.adata_train, desc="Processing train embeddings")
        self.adata_val.obsm['pert_emb'] = self._compute_perturbation_embeddings(self.adata_val, desc="Processing val embeddings")
        self.adata_test.obsm['pert_emb'] = self._compute_perturbation_embeddings(self.adata_test, desc="Processing test embeddings")
        # --- 4. 데이터 정규화 및 DataLoader 생성 (헬퍼 함수 호출) ---
        # .layers['x'] (교란 전) / .layers['y'] (교란 후) 사용
        self._prepare_data_for_training()
        self._calculate_ground_truth_deltas()

        '''
        for i in tqdm(np.unique(self.adata.obs['condition'].values)):
            # 'TP53~C135Y+ctrl' -> 'TP53~C135Y', 'ctrl' -> 'ctrl'
            genes = i.split('+')
            
            # 임베딩을 합산할 변수 초기화
            pert_emb_p = np.zeros(self.p_dim)
            
            # 분리된 유전자(돌연변이 포함) 정보를 순회
            for gene_info in genes:
                if gene_info == 'ctrl':
                    # 'ctrl'은 임베딩을 합산하지 않고 건너뜁니다.
                    continue
                
                # 'TP53~C135Y'를 'TP53'과 'C135Y'로 분리
                gene_name, mutation = gene_info.split('~')
                alt_key = (gene_name, mutation)
                ref_key = (gene_name, 'REF') # REF 키 정의

                # emb_type에 따라 계산 방식 분기
                if self.emb_type == 'ALT':
                    if alt_key in self.gene_emb:
                        pert_emb_p += self.gene_emb[alt_key]
                    else:
                        print(f"Warning: ALT embedding not found for {alt_key}. Skipping in condition '{i}'.")
                
                elif self.emb_type == 'REF-ALT':
                    # REF와 ALT 임베딩이 모두 존재하는지 확인
                    if alt_key in self.gene_emb and ref_key in self.gene_emb:
                        pert_emb_p += (self.gene_emb[ref_key] - self.gene_emb[alt_key]) # 차이 계산
                    else:
                        if alt_key not in self.gene_emb:
                            print(f"Warning: ALT embedding not found for {alt_key}. Skipping in condition '{i}'.")
                        if ref_key not in self.gene_emb:
                            print(f"Warning: REF embedding not found for {ref_key}. Skipping REF-ALT calculation for '{i}'.")
                
                # 딕셔너리 키로 사용할 튜플 생성
                #emb_key = (gene_name, mutation)
                
                # 해당 키의 임베딩을 가져와 합산
                # self.gene_emb은 이미 (1280,) 형태의 임베딩을 갖고 있어야 함
                #pert_emb_p += self.gene_emb[emb_key]
                

            # 계산된 임베딩을 해당 조건의 모든 세포에 할당
            self.pert_emb_cells[self.adata.obs['condition'].values == i] = pert_emb_p
            
            # 섭동 조건별 임베딩을 딕셔너리에 저장
            self.pert_emb[i] = pert_emb_p

        self.adata.obsm['pert_emb'] = self.pert_emb_cells
        

        # control cells
        ctrl_x = adata[adata.obs['condition'].values == 'ctrl'].X
        self.ctrl_mean = np.mean(ctrl_x, axis=0)    # 전체 유전자에 대한 대조군 평균 발현량
        self.ctrl_x = torch.from_numpy(ctrl_x - self.ctrl_mean.reshape(1, -1)).float().to(self.device)
        self.adata.X = self.adata.X - self.ctrl_mean.reshape(1, -1)

        # split datasets
        print("Spliting data...")
        self.adata_train = self.adata[self.adata.obs[split_name].values == 'train']
        self.adata_val = self.adata[self.adata.obs[split_name].values == 'valid']
        self.adata_test = self.adata[self.adata.obs[split_name].values == 'test']
        self.pert_val = np.unique(self.adata_val.obs['condition'].values)

        self.train_data = PertDataset(torch.from_numpy(self.adata_train.X).float().to(self.device), 
                                      torch.from_numpy(self.adata_train.obsm['pert_emb']).float().to(self.device))
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        self.test_data = PertDataset(torch.from_numpy(self.adata_test.X).float().to(self.device), 
                                      torch.from_numpy(self.adata_test.obsm['pert_emb']).float().to(self.device))
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

        
        self.pert_delta = {}
        for i in np.unique(self.adata.obs['condition'].values):
            adata_i = self.adata[self.adata.obs['condition'].values == i]
            delta_i = np.mean(adata_i.X, axis=0)
            self.pert_delta[i] = delta_i
        '''
    def _compute_perturbation_embeddings(self, adata, desc="Processing"):
        """
        주어진 AnnData의 'condition'을 기반으로 교란 임베딩을 계산합니다.
        self.emb_type에 따라 'ALT' 또는 'REF-ALT' 로직을 사용합니다.
        계산된 임베딩은 self.pert_emb 딕셔너리를 업데이트하고, numpy 배열로 반환합니다.
        """
        pert_emb_cells = np.zeros((adata.shape[0], self.p_dim))
        
        for i in tqdm(np.unique(adata.obs['condition'].values), desc=desc):
            # 이미 계산된 임베딩이 있으면 재사용 (효율성)
            if i in self.pert_emb:
                pert_emb_cells[adata.obs['condition'].values == i] = self.pert_emb[i]
                continue

            # 새 임베딩 계산
            genes = i.split('+')
            pert_emb_p = np.zeros(self.p_dim)
            for gene_info in genes:
                if gene_info == 'ctrl': continue
                
                gene_name, mutation = gene_info.split('~')
                alt_key = (gene_name, mutation)
                ref_key = (gene_name, 'REF')

                if self.emb_type == 'ALT':
                    if alt_key in self.gene_emb: 
                        pert_emb_p += self.gene_emb[alt_key]
                    else:
                        print(f"Warning: ALT embedding not found for {alt_key}. Skipping in condition '{i}'.")
                elif self.emb_type == 'REF-ALT':
                    if alt_key in self.gene_emb and ref_key in self.gene_emb:
                        # (참고: REF - ALT가 아닌 ALT - REF로 수정했습니다. 필요시 다시 수정하세요.)
                        pert_emb_p += (self.gene_emb[alt_key] - self.gene_emb[ref_key]) 
                    else:
                        if alt_key not in self.gene_emb:
                            print(f"Warning: ALT embedding not found for {alt_key}. Skipping in condition '{i}'.")
                        if ref_key not in self.gene_emb:
                            print(f"Warning: REF embedding not found for {ref_key}. Skipping REF-ALT calculation for '{i}'.")
            
            pert_emb_cells[adata.obs['condition'].values == i] = pert_emb_p
            self.pert_emb[i] = pert_emb_p # 전역 딕셔너리에 저장
            
        return pert_emb_cells
    def _prepare_data_for_training(self):
        """
        .layers['y'](교란 후)와 .layers['x'](교란 전)를 사용해 델타(변화량)를 계산하고 .X에 저장합니다.
        Train/Test DataLoader를 생성하고, Validation용 텐서를 준비합니다.
        """
        print("Calculating deltas (y - x) and preparing DataLoaders...")
        
        # 1. 델타 계산 (모델 학습 타겟)
        self.adata_train.X = self.adata_train.layers['y'] - self.adata_train.layers['x']
        self.adata_val.X = self.adata_val.layers['y'] - self.adata_val.layers['x']
        
        # 2. DataLoader 생성
        self.train_data = PertDataset(
            torch.from_numpy(self.adata_train.X).float().to(self.device), # .X = 델타
            torch.from_numpy(self.adata_train.obsm['pert_emb']).float().to(self.device)
        )
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)



        # 3. 검증(Validation)용 텐서 준비
        self.val_ctrl_x_tensor = torch.from_numpy(self.adata_val.layers['x']).float().to(self.device)
        # 검증에 사용할 '교란' 임베딩
        self.val_pert_emb_tensor = torch.from_numpy(self.adata_val.obsm['pert_emb']).float().to(self.device)
        # 검증용 고유 교란 목록
        self.pert_val = np.unique(self.adata_val.obs['condition'].values)
        
        self.ctrl_mean_for_gen = np.mean(self.adata_test.layers['x'], axis=0)
        
    def _calculate_ground_truth_deltas(self):
        """
        모든 데이터셋(train, val, test)의 델타(.X) 값을 취합하여
        각 교란 조건별 평균 델타("정답지")를 self.pert_delta에 저장합니다.
        """
        print("Calculating ground truth deltas (mean of deltas)...")
        self.pert_delta = {}
        
        # .obs와 .X(델타)를 모든 데이터셋에서 결합
        all_adata_obs = pd.concat([self.adata_train.obs, self.adata_val.obs, self.adata_test.obs])
        all_adata_delta = np.concatenate([
            self.adata_train.X, 
            self.adata_val.X, 
            self.adata_test.X
        ], axis=0)
        
        for i in tqdm(np.unique(all_adata_obs['condition'].values), desc="Calculating deltas"):
            indices = all_adata_obs['condition'].values == i
            if np.sum(indices) > 0:
                delta_i = np.mean(all_adata_delta[indices], axis=0)
                self.pert_delta[i] = delta_i
    

    def loss_function(self, x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, T):
        reconstruction_loss = 0.5 * torch.mean(torch.sum((x_hat - x)**2, axis=1)) + 0.5 * torch.mean(torch.sum((p_hat - p)**2, axis=1))
        KLD_z = - 0.5 * torch.mean(torch.sum(1 + log_var_z - mean_z**2 - log_var_z.exp(), axis=1))
        MI_latent = torch.mean(T(mean_z, s.detach())) - torch.log(torch.mean(torch.exp(T(mean_z, s_marginal.detach()))))
        return reconstruction_loss + KLD_z + self.lambda_MI * MI_latent

    def loss_recon(self, x, x_hat):
        reconstruction_loss = 0.5 * torch.mean(torch.sum((x_hat - x)**2, axis=1))
        return reconstruction_loss

    def loss_MINE(self, mean_z, s, s_marginal, T):
        MI_latent = torch.mean(T(mean_z, s)) - torch.log(torch.mean(torch.exp(T(mean_z, s_marginal))))
        return - MI_latent

    def train(self, retrain=False):
        if not retrain:
            self.Net = Net(x_dim = self.x_dim, p_dim = self.p_dim, 
                           latent_dim = self.latent_dim, hidden_dim = self.hidden_dim)
        params = list(self.Net.Encoder_x.parameters())+list(self.Net.Encoder_p.parameters())+list(self.Net.Decoder_x.parameters())+list(self.Net.Decoder_p.parameters())
        optimizer = Adam(params, lr=0.0005)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
        optimizer_MINE = Adam(self.Net.MINE.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler_MINE = StepLR(optimizer_MINE, step_size=30, gamma=0.2)

        corr_val_best = 0
        # <<<--- 손실 기록 리스트 추가 ---<<<
        train_loss_history = []
        val_loss_history = []
        # >>>-------------------------->>>
        if retrain:
            if len(self.pert_val) > 0: # If validating
                self.Net.eval()
                corr_ls = []
                for i in self.pert_val:
                    '''
                    if self.multi_gene:
                        genes = i.split('+')
                        pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
                    else:
                        pert_emb_p = self.pert_emb[i]
                    '''
                    if i not in self.pert_emb:
                        print(f"Warning: Pre-calculated embedding not found for validation condition '{i}'. Skipping.")
                        continue
                    pert_emb_p = self.pert_emb[i]
                    val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                                     (self.ctrl_x.shape[0], 1))).float().to(self.device)
                    x_hat, p_hat, mean_z, log_var_z, s = self.Net(self.ctrl_x, val_p)
                    x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0)
                    corr = np.corrcoef(x_hat, self.pert_delta[i])[0, 1]
                    corr_ls.append(corr)

                corr_val_best = np.mean(corr_ls)
                print("Previous best validation correlation delta %.5f" % corr_val_best)
        self.Net.train()
        for epoch in tqdm(range(self.training_epochs)):
            # <<<--- 학습 손실 계산 변수 ---<<<
            epoch_train_loss = 0.0
            batch_count = 0
            # >>>-------------------------->>>
            for x, p in self.train_dataloader:
                # adversarial training on p
                p.requires_grad = True 
                self.Net.eval()
                with torch.enable_grad():
                    x_hat, _, _, _, _ = self.Net(x, p)
                    recon_loss = self.loss_recon(x, x_hat)
                    grads = torch.autograd.grad(recon_loss, p)[0]
                    p_ae = p + self.eps * torch.norm(p, dim=1).view(-1, 1) * torch.sign(grads.data) # generate adversarial examples

                self.Net.train()
                x_hat, p_hat, mean_z, log_var_z, s = self.Net(x, p_ae)

                # for MINE
                index_marginal = np.random.choice(np.arange(len(self.train_data)), size=x_hat.shape[0])
                p_marginal = self.train_data.p[index_marginal]
                s_marginal = self.Net.Encoder_p(p_marginal)
                for _ in range(1):
                    optimizer_MINE.zero_grad()
                    loss = self.loss_MINE(mean_z, s, s_marginal, T=self.Net.MINE)
                    loss.backward(retain_graph=True)
                    optimizer_MINE.step()

                optimizer.zero_grad()
                loss = self.loss_function(x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, T=self.Net.MINE)
                loss.backward()
                optimizer.step()

                # <<<--- 학습 손실 누적 ---<<<
                epoch_train_loss += loss.item()
                batch_count += 1
                # >>>-------------------------->>>
            scheduler.step()
            scheduler_MINE.step()
            # <<<--- 학습 손실 평균 계산 ---<<<
            avg_epoch_loss = epoch_train_loss / batch_count
            # --- W&B 로깅을 위한 딕셔너리 준비 ---
            log_dict = {"epoch": epoch + 1, "train_loss": avg_epoch_loss}
            train_loss_history.append(avg_epoch_loss)
            # >>>-------------------------->>>
            if (epoch+1) % 10 == 0:
                print("\tEpoch", (epoch+1), "complete!", "\t Loss: ", loss.item())
                if len(self.pert_val) > 0: # If validating
                    self.Net.eval()
                    corr_ls = []
                    # <<<--- 검증 손실 기록 리스트 ---<<<
                    val_loss_ls = []
                    # Loop through the validation perturbations
                    with torch.no_grad():
                        for i in self.pert_val:
                            '''
                            # Initialize the perturbation embedding vector with zeros
                            pert_emb_p = np.zeros(self.p_dim)
                            
                            # Split the condition string by '+' to handle both single and dual perturbations
                            gene_conditions = i.split('+')
                            
                            # Check if the condition is not a single 'ctrl'
                            if gene_conditions[0] != 'ctrl':
                                # Iterate through each part of the condition (e.g., 'TP53~A276V', 'ctrl')
                                for gene_info in gene_conditions:
                                    # Skip 'ctrl' parts if they are present in a dual perturbation
                                    if gene_info == 'ctrl':
                                        continue
                                    
                                    # Split the gene and mutation part by '~'
                                    gene_name, mutation = gene_info.split('~')
                                    
                                    # Retrieve the embedding from the dictionary using the tuple key and add it
                                    # self.gene_emb must be a dictionary with keys like ('TP53', 'A276V')
                                    pert_emb_p += self.gene_emb[(gene_name, mutation)]
                            '''        
                            if i not in self.pert_emb:
                                print(f"Warning: Pre-calculated embedding not found for validation condition '{i}'. Skipping.")
                                continue
                            pert_emb_p = self.pert_emb[i]
                            # Tile the perturbation embedding to match the batch size of control cells
                            val_p = torch.from_numpy(np.tile(pert_emb_p, (self.ctrl_x.shape[0], 1))).float().to(self.device)
                            
                            # Pass the data through the model
                            x_hat_val, p_hat_val, mean_z_val, log_var_z_val, s_val = self.Net(self.ctrl_x, val_p)
                            
                            # 검증 손실 계산
                            index_marginal_val = np.random.choice(np.arange(len(self.train_data)), size=self.ctrl_x.shape[0])
                            p_marginal_val = self.train_data.p[index_marginal_val].to(self.device) 
                            s_marginal_val = self.Net.Encoder_p(p_marginal_val)
                            val_loss = self.loss_function(self.ctrl_x, x_hat_val, val_p, p_hat_val, mean_z_val, log_var_z_val, s_val, s_marginal_val, T=self.Net.MINE)
                            # <<<--- 핵심 수정 사항: .item() 추가 ---<<<
                            val_loss_ls.append(val_loss.item()) 
                            # >>>---------------------------------->>>
                            
                            # 상관계수 계산
                            x_hat_mean_val = np.mean(x_hat_val.cpu().numpy(), axis=0)
                            corr = np.corrcoef(x_hat_mean_val, self.pert_delta[i])[0, 1]
                            corr_ls.append(corr)
                    # <<<--- 평균 검증 손실 계산 ---<<<
                    avg_val_loss = np.mean(val_loss_ls) if val_loss_ls else float('nan')
                    val_loss_history.append(avg_val_loss)
                    # >>>-------------------------->>>
                    # Calculate and print the average validation correlation
                    corr_val = np.mean(corr_ls)
                    print("Validation correlation delta %.5f" % corr_val)

                    # --- W&B 로깅 딕셔너리에 검증 결과 추가 ---
                    log_dict["val_loss"] = avg_val_loss
                    log_dict["val_correlation"] = corr_val
                    # ----------------------------------------
                    
                    # Save the best model based on validation correlation
                    if corr_val > corr_val_best:
                        corr_val_best = corr_val
                        self.model_best = copy.deepcopy(self.Net)
                    
                    # Set the model back to training mode
                    self.Net.train()

                else:
                    # If no validation is performed, save the final model
                    if epoch == (self.training_epochs-1):
                        self.model_best = copy.deepcopy(self.Net)
            # --- 매 에포크마다 W&B에 로깅 ---
            if self.wandb_run:
                self.wandb_run.log(log_dict)
            # -------------------------------
        print("Finish training.")
        self.Net = self.model_best
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'Net': self.Net.state_dict()}
        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))

    def load_pretrain(self):
        self.Net = Net(x_dim = self.x_dim, p_dim = self.p_dim, 
                       latent_dim = self.latent_dim, hidden_dim = self.hidden_dim)
        self.Net.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['Net'])
    '''
    def predict(self, 
                pert_test, # perturbation or a list of perturbations
                return_type = 'cells' # return mean or cells
                ):
        self.Net.eval()
        res = {} 
         # Ensure pert_test is always a list
        if isinstance(pert_test, str):
            pert_test = [pert_test]
        
        for i in pert_test:
            # __init__에서 계산된 임베딩을 직접 가져옴 (ALT 또는 REF-ALT 결과)
            if i not in self.pert_emb:
                print(f"Warning: Pre-calculated embedding not found for prediction condition '{i}'. Skipping.")
                continue
            pert_emb_p = self.pert_emb[i]
            # Tile the perturbation embedding to match the batch size of control cells
            val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                            (self.ctrl_x.shape[0], 1))).float().to(self.device)
            
            # Pass the control cells and perturbation embedding through the model
            x_hat, p_hat, mean_z, log_var_z, s = self.Net(self.ctrl_x, val_p)
            
            # Format the output based on the return_type
            if return_type == 'cells':
                # Add the control mean back to get the absolute expression values
                adata_pred = ad.AnnData(X=(x_hat.detach().cpu().numpy() + self.ctrl_mean.reshape(1, -1)))
                adata_pred.obs['condition'] = i
                res[i] = adata_pred
            elif return_type == 'mean':
                x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0) + self.ctrl_mean
                res[i] = x_hat
            else:
                raise ValueError("return_type can only be 'mean' or 'cells'.")
        
        return res
    '''
    def predict(self, 
                pert_test, # A list of condition names from the test set
                return_type = 'cells' 
                ):
        self.Net.eval()
        res = {} 
        
        if isinstance(pert_test, str):
            pert_test = [pert_test]
        
        # __init__에서 저장해 둔 self.adata_test를 사용
        
        for i in pert_test: # 'TP53~A276V+ctrl'와 같은 각 교란 조건에 대해 반복
            
            # 1. 테스트 세트(self.adata_test)에서 이 조건에 해당하는 세포들만 찾기
            condition_mask = self.adata_test.obs['condition'] == i
            if not np.any(condition_mask):
                print(f"Warning: No cells found for prediction condition '{i}' in self.adata_test. Skipping.")
                continue
                
            # 2. 이 세포들의 '교란 전' 상태(.layers['x'])를 가져옴
            ctrl_x_tensor = torch.from_numpy(self.adata_test.layers['x'][condition_mask]).float().to(self.device)
            
            # 3. 이 교란 조건에 해당하는 임베딩을 가져옴
            if i not in self.pert_emb:
                print(f"Warning: Pre-calculated embedding not found for '{i}'. Skipping.")
                continue
            pert_emb_p = self.pert_emb[i]
            
            # 4. 교란 임베딩을 2번에서 찾은 세포 수만큼 복제
            pred_p = torch.from_numpy(np.tile(pert_emb_p, 
                                            (ctrl_x_tensor.shape[0], 1))).float().to(self.device)
            
            # 5. 모델을 사용해 '변화량(delta)' 예측
            with torch.no_grad():
                # 입력: (테스트용 교란 전 상태, 교란 임베딩), 출력: 예측된 델타
                x_hat_delta, _, _, _, _ = self.Net(ctrl_x_tensor, pred_p)

            # 6. '변화량'을 '최종 발현량'으로 변환
            # 최종 발현량 = 예측된 델타 + 원본 교란 전 상태
            final_expression = x_hat_delta.detach().cpu().numpy() + self.adata_test.layers['x'][condition_mask]

            # 7. 결과 저장
            if return_type == 'cells':
                adata_pred = ad.AnnData(X=final_expression)
                adata_pred.obs['condition'] = i
                res[i] = adata_pred
                
            elif return_type == 'mean':
                x_hat_mean = np.mean(final_expression, axis=0)
                res[i] = x_hat_mean
            else:
                raise ValueError("return_type can only be 'mean' or 'cells'.")
        
        return res

    def generate(self, 
                 pert_test, # perturbation or a list of perturbations
                 return_type = 'cells', # return mean or cells
                 n_cells = 10000 # number of cells to generate
                 ):
        self.Net.eval()
        res = {} 
        # Ensure pert_test is always a list
        if isinstance(pert_test, str):
            pert_test = [pert_test]
        
        for i in pert_test:
            '''
            # Initialize the perturbation embedding vector
            pert_emb_p = np.zeros(self.p_dim)
            
            # Split the condition string (e.g., 'TP53~C135Y+ctrl' -> ['TP53~C135Y', 'ctrl'])
            gene_conditions = i.split('+')
            
            # Check for the 'ctrl' condition which has no associated embedding
            if gene_conditions[0] != 'ctrl':
                # Iterate through each gene/mutation part of the condition
                for gene_info in gene_conditions:
                    # Skip the 'ctrl' part if it's in a dual perturbation
                    if gene_info == 'ctrl':
                        continue
                    
                    # Split the gene-mutation string (e.g., 'TP53~C135Y' -> 'TP53', 'C135Y')
                    gene_name, mutation = gene_info.split('~')
                    
                    # Retrieve the embedding from the dictionary using a tuple key and sum it
                    # Assumes self.gene_emb has keys like ('TP53', 'C135Y')
                    pert_emb_p += self.gene_emb[(gene_name, mutation)]
            '''
            if i not in self.pert_emb:
                print(f"Warning: Pre-calculated embedding not found for generation condition '{i}'. Skipping.")
                continue
            pert_emb_p = self.pert_emb[i]
            # Tile the perturbation embedding to match the number of cells to generate
            val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                            (n_cells, 1))).float().to(self.device)
            
            # Pass the perturbation embedding through the model's encoder
            s = self.Net.Encoder_p(val_p)
            
            # Generate random noise for the latent space
            z = torch.randn(n_cells, self.latent_dim).to(self.device)
            
            # Combine noise and perturbation effect and pass through the decoder to generate data
            x_hat = self.Net.Decoder_x(z + s)
            
            # Format the output based on the return_type
            if return_type == 'cells':
                # Add the control mean back to get the absolute expression values
                res[i] = x_hat.detach().cpu().numpy() + self.ctrl_mean.reshape(1, -1)
            elif return_type == 'mean':
                x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0) + self.ctrl_mean
                res[i] = x_hat
            else:
                raise ValueError("return_type can only be 'mean' or 'cells'.")
        
        return res

    def get_embedding(self, adata=None):
        if adata == None:
            input_adata = None
            adata = self.adata
        x = torch.from_numpy(adata.X).float().to(self.device)
        p = torch.from_numpy(adata.obsm['pert_emb']).float().to(self.device)
        x_hat, p_hat, mean_z, log_var_z, s = self.Net(x, p)
        adata.obsm['mean_z'] = mean_z.detach().cpu().numpy()
        adata.obsm['z+s'] = adata.obsm['mean_z'] + s.detach().cpu().numpy()

        emb_s = pd.DataFrame(s.detach().cpu().numpy(), index=adata.obs['condition'].values)
        emb_s = emb_s.groupby(emb_s.index, axis=0).mean()
        adata.uns['emb_s'] = emb_s
        if input_adata is None:
            self.adata = adata
        return adata



class PertDataset(Dataset):
    def __init__(self, x, p):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x = x
        self.p = p

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(self.device), self.p[idx].to(self.device)