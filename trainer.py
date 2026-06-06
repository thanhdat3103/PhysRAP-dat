import math
import torch, os
import numpy as np

import random
import time
import argparse
import csv
import logging
import shutil

from copy import deepcopy
from scipy import io as sio
from scipy import signal
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.engine import build_dataset, build_optimizer, build_scheduler, build_criterion, build_model
from utils.util import AvgrageMeter, pearson_correlation_coefficient, update_avg_meters, cal_psd_hr, \
    augment_flip, augment_time_reversal, random_resized_crop, augment_gaussian_noise


def set_seed(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class RppgEstimatorTrainer:
    def __init__(self, args) -> None:
        self.args = args

        self.gpu_list = [int(i) for i in args.gpu.split(',')]
        self.gpu_num = len(self.gpu_list)
        self.actual_batch_size = args.batch_size * self.gpu_num
        self.device = torch.device(f'cuda:{self.gpu_list[0]}')

        self.rppg_estimator_stu = build_model(args).to(self.device)
        self.rppg_estimator_teacher = build_model(args).to(self.device)

        ## generate save path
        self.run_date = time.strftime('%m%d_%H%M', time.localtime(time.time()))
        self.current_date = self.run_date
        self.save_path = f'{args.save_path}/{self.run_date}'

        ## dataloader NOTE: SELECT YOUR DATASET
        self.all_datasets = self.args.datasets.split('_')  # VIPL_BUAA_UBFC_PURE' => ['VIPL', 'BUAA', 'UBFC', 'PURE']
        self.train_dataloaders = build_dataset(args, mode='train_all', batch_size=self.actual_batch_size)
        self.val_dataloaders = build_dataset(args, mode='test_all', batch_size=1)

        ## optimizer
        self.optimizer = build_optimizer(args, self.rppg_estimator_stu)
        self.scheduler = build_scheduler(args, self.optimizer)
        self.loss_funcs = build_criterion(args)
        self.loss_funcs_weight = dict(zip(eval(args.loss), eval(args.loss_weight)))

        ## loss & metrics saver
        self.loss_meters = dict([(key, AvgrageMeter()) for key in self.loss_funcs.keys()])
        self.metrics_meters = {
            'mae': AvgrageMeter(),
        }

        ## constant
        self.bpm_range = torch.arange(40, 180, dtype=torch.float).to(self.device)
        self.best_epoch = 0
        self.best_val_mae = 1000    # mean absolute error
        self.best_val_rmse = 1000   # root mean square error
        self.best_sd = 1000         # standard deviation
        self.best_r = 0             # Pearson correlation coefficient
        self.frame_rate = 30

    def prepare_train(self, start_dataset_idx, continue_log):
        """Prepares the training process."""
        if start_dataset_idx != 0:
            self.save_path = f'{self.args.save_path}/{continue_log}'
            self.run_date = continue_log

        self.save_ckpt_path = f'{self.save_path}/ckpt'
        self.save_rppg_path = f'{self.save_path}/rppg'
        if not os.path.exists(self.save_ckpt_path):
            os.makedirs(self.save_ckpt_path)
        if not os.path.exists(self.save_rppg_path):
            os.makedirs(self.save_rppg_path)

        all_dataset_first_name = ''.join([i[0] for i in self.all_datasets])
        logging.basicConfig(
            filename=f'./logs/{self.args.model}_{all_dataset_first_name}_{self.args.num_rppg}_S{self.run_date}_N{self.current_date}.log',
            format='%(message)s',
            filemode='a'
        )
        self.logger = logging.getLogger(
            f'./logs/{self.args.model}_{all_dataset_first_name}_{self.args.num_rppg}_S{self.run_date}_N{self.current_date}'
        )
        self.logger.setLevel(logging.INFO)

        ## save proj_file to save_path
        cur_file = os.getcwd()
        cur_file_name = cur_file.split('/')[-1]
        shutil.copytree(f'{cur_file}', f'{self.save_path}/{self.current_date}/{cur_file_name}', dirs_exist_ok=True)

        if start_dataset_idx != 0:
            if not os.path.exists(f'{self.save_ckpt_path}/rppg_estimator_stu_dataset_{start_dataset_idx-1}.pth'):
                raise Exception(f'rppg_estimator_stu ckpt file {start_dataset_idx-1} not found')
            self.rppg_estimator_stu.load_state_dict(
                torch.load(
                    f'{self.save_ckpt_path}/rppg_estimator_stu_dataset_{start_dataset_idx-1}.pth',
                    map_location=self.device
                )
            )
            self.rppg_estimator_teacher.load_state_dict(
                torch.load(
                    f'{self.save_ckpt_path}/rppg_estimator_teacher_dataset_{start_dataset_idx-1}.pth',
                    map_location=self.device
                )
            )

        print(f'save_path: {self.save_path}, log_path: ./logs/{self.args.model}_{self.args.datasets}_{self.args.num_rppg}_{self.run_date}')

        ## block gradient and set train
        self.rppg_estimator_stu.train()
        self.rppg_estimator_teacher.eval()

    def draw_rppg_ecg(self, rPPG, ecg, save_path_epoch, train=False, mini_batch=0):
        """Draws rPPG and ECG signals, saves the results, and plots the power spectral density."""
        rPPG_sample, ecg_sample = rPPG[0], ecg[0]

        ## save the results
        b, a = signal.butter(2, [0.67 / 15, 3 / 15], 'bandpass')
        rPPG_np = rPPG_sample.cpu().data.numpy()
        rPPG_np = signal.lfilter(b, a, rPPG_np)
        y1 = rPPG_np
        y2 = ecg_sample.cpu().data.numpy()
        results_rPPG = [y1, y2]

        if not train:
            sio.savemat(
                os.path.join(save_path_epoch, 'test_rPPG.mat'),
                {'results_rPPG': results_rPPG},
            )
        else:
            sio.savemat(
                os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.mat'),
                {'results_rPPG': results_rPPG}
            )

        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        psd_pred = cal_psd_hr(rPPG_sample, self.frame_rate, return_type='psd')
        psd_gt = cal_psd_hr(ecg_sample, self.frame_rate, return_type='psd')
        ax[0].set_title('rPPG')
        ax[0].plot(y1, label='rPPG')
        ax[0].plot(y2, label='ecg')
        ax[0].legend()
        ax[1].set_title('psd')
        ax[1].plot(psd_pred.cpu().data.numpy(), label='pred')
        ax[1].plot(psd_gt.cpu().data.numpy(), label='gt')
        ax[1].legend()

        if not train:
            fig.savefig(os.path.join(save_path_epoch, 'test_rPPG.jpg'))
        else:
            fig.savefig(os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.jpg'))
        plt.close(fig)

    def update_best(self, epoch, hr_pred, hr_gt, val_type='video'):
        """Updates the best validation metrics and saves the model if the current metrics are better."""
        cur_mae = np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))
        cur_rmse = np.sqrt(np.mean(np.square(np.array(hr_gt) - np.array(hr_pred))))
        cur_sd = np.std(np.array(hr_gt) - np.array(hr_pred))
        cur_r = pearson_correlation_coefficient(np.array(hr_gt), np.array(hr_pred))

        self.logger.info(f'evaluate epoch {epoch}, total val {len(hr_gt)} ----------------------------------')
        self.logger.info(f'{val_type}-level mae of model: {np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))}')
        self.logger.info(f'{val_type}-level cur mae: {cur_mae:.2f}, cur rmse: {cur_rmse:.2f}, cur sd: {cur_sd:.2f}, cur r: {cur_r:.4f}')
        self.logger.info(
            f'{val_type}-level best mae of model: {self.best_val_mae:.2f}, best rmse: {self.best_val_rmse:.2f}, best epoch: {self.best_epoch}, '
            f'best sd: {self.best_sd:.2f}, best r: {self.best_r:.4f}'
        )
        self.logger.info('------------------------------------------------------------------')

        return cur_mae, cur_rmse, cur_sd, cur_r

    def evaluate_clip(self, epoch=0, val_dataloader=None):
        """Evaluates the clip data and saves the results."""
        save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
        hr_gt = []
        hr_pred = []

        with torch.no_grad():
            for sample_batched in tqdm(val_dataloader):
                inputs = sample_batched['video'].to(self.device)
                ecg = sample_batched['ecg'].to(self.device)
                clip_average_HR = sample_batched['clip_avg_hr'].to(self.device)

                num_clip = 3
                input_len = inputs.shape[2]
                input_len = input_len - input_len % (num_clip * 4)
                clip_len = input_len // num_clip

                inputs = inputs[:, :, :input_len, :, :]
                ecg = ecg[:, :input_len]

                new_args = deepcopy(self.args)
                new_args.num_rppg = clip_len
                val_rppg_estimator = build_model(new_args).to(self.device)
                val_rppg_estimator.load_state_dict(
                    torch.load(
                        f'{self.save_ckpt_path}/rppg_estimator_stu_epoch_{epoch}.pth',
                        map_location=self.device
                    )
                )
                val_rppg_estimator.eval()

                psd_gt_total = 0
                psd_pred_total = 0

                for idx in range(num_clip):
                    inputs_iter = inputs[:, :, idx * clip_len:(idx + 1) * clip_len, :, :]
                    ecg_iter = ecg[:, idx * clip_len:(idx + 1) * clip_len]

                    psd_gt = cal_psd_hr(ecg_iter, self.frame_rate, return_type='psd')
                    psd_gt_total += psd_gt.view(-1).max(0)[1].cpu() + 40

                    outputs = val_rppg_estimator({'input_clip': inputs_iter})
                    rPPG = outputs['rPPG']

                    psd_pred = cal_psd_hr(rPPG[0], self.frame_rate, return_type='psd')
                    psd_pred_total += psd_pred.view(-1).max(0)[1].cpu() + 40

                hr_pred.append(float((psd_pred_total / num_clip).item()))

                if self.args.eval_gt_mode == 'label':
                    # Use the stored HDF5 gt_hr / clip_avg_hr label as evaluation GT.
                    # This is also the HR target used by ce_loss during training.
                    hr_gt.append(float(clip_average_HR.detach().view(-1).float().mean().cpu().item()))
                else:
                    # Original diagnostic mode: recompute GT HR from ECG PSD.
                    hr_gt.append(float((psd_gt_total / num_clip).item()))

        self.draw_rppg_ecg(rPPG, ecg_iter, save_path_epoch)
        return self.update_best(epoch, hr_pred, hr_gt, val_type='clip')    

    def initial_train_one_epoch(self, epoch, save_path_epoch, train_dataloader):
        with tqdm(range(len(train_dataloader))) as pbar:
            for iter_idx, sample_batched in zip(pbar, train_dataloader):
                inputs, ecg, clip_average_HR = sample_batched['video'].to(self.device), \
                    sample_batched['ecg'].to(self.device), sample_batched['clip_avg_hr'].to(self.device)

                self.optimizer.zero_grad()

                all_inputs = {
                    'input_clip': inputs,
                    'gra_sharp': 2.0
                }
                outputs = self.rppg_estimator_stu(all_inputs)
                rPPG = outputs['rPPG']

                train_losses = {}
                train_losses['np_loss'] = self.loss_funcs['np_loss'](rPPG, ecg)

                fre_loss, kl_loss, train_mae = self.loss_funcs['ce_loss'](rPPG, clip_average_HR)
                train_losses['ce_loss'] = fre_loss + kl_loss

                total_loss = sum(
                    train_losses[key] * self.loss_funcs_weight[key]
                    for key in train_losses
                )
                total_loss.backward()
                self.optimizer.step()

                train_metrics = {
                    'mae': train_mae,
                }
                update_avg_meters(self.loss_meters, train_losses, self.actual_batch_size)
                update_avg_meters(self.metrics_meters, train_metrics, self.actual_batch_size)

                mini_batch_info = f'epoch : {epoch:0>3}, mini-batch : {iter_idx:0>4}, lr = {self.optimizer.param_groups[0]["lr"]:.5f}'
                loss_info = ', '.join([f'{key} = {self.loss_meters[key].avg:.4f}' for key in self.loss_meters.keys()])
                metrics_info = ', '.join([f'{key} = {self.metrics_meters[key].avg:.4f}' for key in self.metrics_meters.keys()])

                if iter_idx % self.args.echo_batches == self.args.echo_batches - 1:
                    self.logger.info(', '.join([mini_batch_info, loss_info, metrics_info]))
                    self.draw_rppg_ecg(rPPG, ecg, save_path_epoch, train=True, mini_batch=iter_idx)

                pbar.set_description(', '.join([mini_batch_info, loss_info, metrics_info]))

        self.scheduler.step()
        return train_losses

    def initial_train(self, dataset_idx):
        for epoch in range(0, self.args.epochs):
            save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
            if not os.path.exists(save_path_epoch):
                os.makedirs(save_path_epoch)
            self.logger.info(f'train epoch: {epoch} lr: {self.optimizer.param_groups[0]["lr"]:.5f}')
            self.initial_train_one_epoch(epoch, save_path_epoch, self.train_dataloaders[dataset_idx])

            torch.save(
                self.rppg_estimator_stu.state_dict(),
                os.path.join(self.save_ckpt_path, f'rppg_estimator_stu_epoch_{epoch}.pth')
            )

        torch.save(
            self.rppg_estimator_stu.state_dict(),
            os.path.join(self.save_ckpt_path, f'rppg_estimator_stu_dataset_{dataset_idx}.pth')
        )
        torch.save(
            self.rppg_estimator_teacher.state_dict(),
            os.path.join(self.save_ckpt_path, f'rppg_estimator_teacher_dataset_{dataset_idx}.pth')
        )

    def continue_tta(self, dataset_idx):
        """TTA the model && cal the metrics."""

        def argumation(inputs):
            # inputs: [B, C, T, H, W]
            # outputs: list of augmented clips
            N = self.args.tta_num_augs

            if self.args.tta_aug_mode == 'identity':
                return [inputs.clone() for _ in range(N)]

            if self.args.tta_aug_mode == 'gaussian_only':
                return [augment_gaussian_noise(inputs) for _ in range(N)]

            if self.args.tta_aug_mode == 'crop_only':
                return [random_resized_crop(inputs) for _ in range(N)]

            if self.args.tta_aug_mode == 'flip_only':
                return [augment_flip(inputs) for _ in range(N)]

            if self.args.tta_aug_mode == 'reverse_only':
                return [augment_time_reversal(inputs) for _ in range(N)]

            if self.args.tta_aug_mode == 'gaussian_crop':
                return [random_resized_crop(augment_gaussian_noise(inputs)) for _ in range(N)]

            aug_videos = []
            available_augs = [
                augment_gaussian_noise,
                random_resized_crop,
                augment_flip,
                augment_time_reversal
            ]
            for _ in range(N):
                aug_videos.append(random.choice(available_augs)(inputs))
            return aug_videos
        
        def hr_bpm_from_rppg_batch(rppg_batch):
            bpm_list = []
            for b in range(rppg_batch.shape[0]):
                psd = cal_psd_hr(rppg_batch[b], self.frame_rate, return_type='psd')
                bpm_list.append((psd.view(-1).max(0)[1] + 40).float())
            return torch.stack(bpm_list, 0)

        LOAD_DATASET = dataset_idx - 1
        self.rppg_estimator_stu.load_state_dict(
            torch.load(
                f'{self.save_ckpt_path}/rppg_estimator_stu_dataset_{LOAD_DATASET}.pth',
                map_location=self.device
            )
        )
        if dataset_idx == 1:
            self.rppg_estimator_teacher.load_state_dict(
                torch.load(
                    f'{self.save_ckpt_path}/rppg_estimator_stu_dataset_{LOAD_DATASET}.pth',
                    map_location=self.device
                )
            )
        else:
            self.rppg_estimator_teacher.load_state_dict(
                torch.load(
                    f'{self.save_ckpt_path}/rppg_estimator_teacher_dataset_{LOAD_DATASET}.pth',
                    map_location=self.device
                )
            )

        self.rppg_estimator_stu.train()
        self.rppg_estimator_teacher.eval()

        ablation_mode = self.args.ablation_mode
        use_rs = ablation_mode in ['full', 'priors_rs']
        use_pa = ablation_mode in ['full', 'priors_pa']

        self.logger.info(
            f'[ABLATION] mode={ablation_mode}, use_rs={use_rs}, use_pa={use_pa}, tta_aug_mode={self.args.tta_aug_mode}'
        )
        print(
            f'[ABLATION] mode={ablation_mode}, use_rs={use_rs}, use_pa={use_pa}, tta_aug_mode={self.args.tta_aug_mode}'
        )

        tta_dataloader = self.val_dataloaders[dataset_idx]
        hr_gt = []
        hr_pred = []
        debug_max_clips = 40
        debug_seen = 0

        for sample_batched in tqdm(tta_dataloader):
            inputs, ecg, clip_average_HR = sample_batched['video'].to(self.device), \
                sample_batched['ecg'].to(self.device), sample_batched['clip_avg_hr'].to(self.device)

            B, C, T, H, W = inputs.shape
            num_clip_per_video = T // self.args.num_rppg
            inputs = inputs[:, :, :num_clip_per_video * self.args.num_rppg, :, :]
            clip_level_inputs = inputs.view(
                B, C, T // self.args.num_rppg, self.args.num_rppg, H, W
            ).permute(2, 0, 1, 3, 4, 5)
            clip_level_inputs = clip_level_inputs[
                :num_clip_per_video // self.args.batch_size * self.args.batch_size
            ]
            clip_level_inputs = clip_level_inputs.view(
                -1, self.args.batch_size, C, self.args.num_rppg, H, W
            )

            for clip_idx, clip_input in enumerate(clip_level_inputs):
                ## Step 1: augmented views from teacher
                augment_inputs = argumation(clip_input)
                augment_psds = []
                augment_rppgs = []
                for augment_input in augment_inputs:
                    output_rppg = self.rppg_estimator_teacher({'input_clip': augment_input})['rPPG']
                    output_psd_all_batch = []
                    for batch_idx in range(augment_input.shape[0]):
                        output_psd = cal_psd_hr(output_rppg[batch_idx], self.frame_rate, return_type='psd')
                        output_psd_all_batch.append(output_psd)
                    output_psds = torch.stack(output_psd_all_batch, 0)
                    augment_psds.append(output_psds)
                    augment_rppgs.append(output_rppg)
                augment_rppgs = torch.stack(augment_rppgs, 0)  # [N, B, T]
                augment_psds = torch.stack(augment_psds, 0)    # [N, B, 140]

                ## Step 2: original student outputs
                origional_rppg = self.rppg_estimator_stu({'input_clip': clip_input})['rPPG']
                origional_psds = []
                for batch_idx in range(clip_input.shape[0]):
                    output_psd = cal_psd_hr(origional_rppg[batch_idx], self.frame_rate, return_type='psd')
                    origional_psds.append(output_psd)
                origional_psds = torch.stack(origional_psds, 0)  # [B, 140]
                hr_before = hr_bpm_from_rppg_batch(origional_rppg)

                ## Step 3: pseudo labels from uncertainty
                all_batch_rppg_uncertainty = []
                for batch_idx in range(clip_input.shape[0]):
                    cur_batch_rppg = augment_rppgs[:, batch_idx, :]
                    cur_batch_rppg = cur_batch_rppg / torch.norm(cur_batch_rppg, p=2, dim=1).unsqueeze(1)
                    cur_batch_origional_rppg = origional_rppg[batch_idx, :]
                    cur_batch_origional_rppg = cur_batch_origional_rppg / torch.norm(cur_batch_origional_rppg, p=2)
                    cur_batch_rppg_diff = cur_batch_rppg - cur_batch_origional_rppg
                    cur_batch_rppg_uncertainty = torch.exp(cur_batch_rppg_diff.mean(1))
                    all_batch_rppg_uncertainty.append(cur_batch_rppg_uncertainty)
                all_batch_rppg_uncertainty = torch.stack(all_batch_rppg_uncertainty, 0)  # [B, N]

                pesudo_label_psd_hr = []
                all_batch_psd_uncertainty = []
                for batch_idx in range(clip_input.shape[0]):
                    cur_batch_psd = augment_psds[:, batch_idx, :]
                    cur_batch_psd = cur_batch_psd / torch.norm(cur_batch_psd, p=2, dim=1).unsqueeze(1)
                    cur_batch_origional_psd = origional_psds[batch_idx, :]
                    cur_batch_origional_psd = cur_batch_origional_psd / torch.norm(cur_batch_origional_psd, p=2)
                    cur_batch_psd_diff = cur_batch_psd - cur_batch_origional_psd
                    cur_batch_psd_uncertainty = torch.exp(cur_batch_psd_diff.mean(1))
                    all_batch_psd_uncertainty.append(cur_batch_psd_uncertainty)
                    pesudo_label_psd_per_batch = cur_batch_psd_uncertainty.unsqueeze(1) * cur_batch_psd
                    pesudo_label_psd_hr.append(pesudo_label_psd_per_batch.mean(0).max(0)[1] + 40)
                all_batch_psd_uncertainty = torch.stack(all_batch_psd_uncertainty, 0)  # [B, N]
                pesudo_label_hr = torch.stack(pesudo_label_psd_hr, 0)
                pseudo_hr = pesudo_label_hr.float()

                ## Step 4: store original params
                origional_params = {
                    k: v.clone() for k, v in self.rppg_estimator_stu.named_parameters()
                }

                ## Step 5: priors by FIM
                uncertainty = all_batch_psd_uncertainty + all_batch_rppg_uncertainty
                self.optimizer.zero_grad()
                uncertainty.mean(0).mean(0).backward()

                with torch.no_grad():
                    fisher_dict = {}
                    for nm, m in self.rppg_estimator_stu.named_modules():
                        for npp, p in m.named_parameters():
                            if npp in ['weight', 'bias'] and p.requires_grad and p.grad is not None:
                                fisher_dict[f"{nm}.{npp}"] = p.grad.data.clone().view(-1)

                    fisher_grads_name = list(fisher_dict.keys())
                    fisher_grads = [fisher_dict[key] for key in fisher_grads_name]

                    fim_matrix = torch.zeros((len(fisher_grads), len(fisher_grads)))
                    for i in range(len(fisher_grads)):
                        for j in range(len(fisher_grads)):
                            param_i, param_j = fisher_grads[i], fisher_grads[j]
                            sim_score = torch.sum(param_i.mean() * param_j.mean())
                            fim_matrix[i, j] = sim_score.item()

                    need_to_update = {}
                    diag_fim = torch.diag(fim_matrix)
                    save_ratio = self.args.tta_save_ratio
                    related_save_ratio = self.args.tta_related_save_ratio

                    # Priors core: select sensitive parameters by diagonal FIM
                    threshold = torch.sort(diag_fim, descending=True)[0][
                        int(diag_fim.shape[0] * (1 - save_ratio))
                    ]
                    for i in range(diag_fim.shape[0]):
                        if diag_fim[i] > threshold:
                            need_to_update[fisher_grads_name[i]] = origional_params[fisher_grads_name[i]]

                    selected_after_priors = len(need_to_update)
                    selected_after_rs = selected_after_priors

                    # RS: relatedness-based pruning using off-diagonal FIM
                    if use_rs:
                        for i in range(diag_fim.shape[0]):
                            name_i = fisher_grads_name[i]
                            if name_i not in need_to_update:
                                continue

                            related_threshold = torch.sort(fim_matrix[i], descending=True)[0][
                                int(diag_fim.shape[0] * related_save_ratio)
                            ]

                            for j in range(diag_fim.shape[0]):
                                name_j = fisher_grads_name[j]
                                if j != i and fim_matrix[i, j] > related_threshold and name_j in need_to_update:
                                    need_to_update.pop(name_j, None)

                    selected_after_rs = len(need_to_update)

                    if clip_idx == 0:
                        self.logger.info(
                            f'[ABLATION] mode={ablation_mode}, selected_after_priors={selected_after_priors}, selected_after_rs={selected_after_rs}'
                        )
                        print(
                            f'[ABLATION] mode={ablation_mode}, selected_after_priors={selected_after_priors}, selected_after_rs={selected_after_rs}'
                        )

                ## Step 6: future gradients for PA
                future_grads = {}
                if use_pa:
                    self.optimizer.zero_grad()
                    K = self.args.tta_future_steps
                    for _ in range(K):
                        selected_inputs = random.choice(augment_inputs)
                        selected_rppg = self.rppg_estimator_stu({'input_clip': selected_inputs})['rPPG']
                        fre_loss, kl_loss, train_mae = self.loss_funcs['ce_loss'](
                            selected_rppg, pesudo_label_hr.detach()
                        )
                        total_loss = fre_loss + kl_loss
                        total_loss.backward()

                    for nm, m in self.rppg_estimator_stu.named_modules():
                        for npp, p in m.named_parameters():
                            if npp in ['weight', 'bias'] and p.requires_grad and p.grad is not None:
                                future_grads[f"{nm}.{npp}"] = p.grad.data.clone().view(-1)

                ## Step 7: current loss + optional PA
                self.optimizer.zero_grad()
                rPPG = self.rppg_estimator_stu({'input_clip': clip_input})['rPPG']
                fre_loss, kl_loss, train_mae = self.loss_funcs['ce_loss'](rPPG, pesudo_label_hr.detach())
                total_loss = fre_loss + kl_loss
                total_loss.backward()

                for nm, m in self.rppg_estimator_stu.named_modules():
                    for npp, p in m.named_parameters():
                        key = f"{nm}.{npp}"
                        if npp in ['weight', 'bias'] and p.requires_grad and key in need_to_update:
                            if use_pa and key in future_grads:
                                current_grad = p.grad.data.clone().view(-1)
                                future_grad = future_grads[key]

                                denom = torch.norm(current_grad) * torch.norm(future_grad) + 1e-12
                                cos_value = torch.dot(current_grad, future_grad) / denom

                                if cos_value > 0:
                                    weight = (
                                        torch.norm(future_grad)
                                        * (cos_value - math.sqrt(2) / 2)
                                        / (torch.norm(current_grad) + 1e-12)
                                    )
                                    weight_exp = 1 / (1 + torch.exp(-weight))
                                    p.grad.data = p.grad.data * weight_exp
                                else:
                                    need_to_update.pop(key, None)

                self.optimizer.step()

                ## Step 8: restore parameters not selected for update
                for nm, m in self.rppg_estimator_stu.named_modules():
                    for npp, p in m.named_parameters():
                        key = f"{nm}.{npp}"
                        if npp in ['weight', 'bias'] and p.requires_grad and key not in need_to_update:
                            mask_fish = torch.ones_like(origional_params[key])
                            mask = mask_fish
                            with torch.no_grad():
                                p.data = origional_params[key] * mask + p * (1. - mask)

                param_delta_sum = 0.0
                param_delta_max = 0.0
                changed_param_count = 0

                for nm, m in self.rppg_estimator_stu.named_modules():
                    for npp, p in m.named_parameters():
                        key = f"{nm}.{npp}"
                        if npp in ['weight', 'bias'] and p.requires_grad and key in need_to_update:
                            delta = torch.norm((p.data - origional_params[key]).view(-1), p=2).item()
                            param_delta_sum += delta
                            param_delta_max = max(param_delta_max, delta)
                            if delta > 1e-12:
                                changed_param_count += 1                

                ## Step 9: EMA update the teacher model
                alpha = self.args.tta_teacher_alpha
                for param_teacher, param_student in zip(
                    self.rppg_estimator_teacher.parameters(),
                    self.rppg_estimator_stu.parameters()
                ):
                    param_teacher.data = alpha * param_teacher.data + (1 - alpha) * param_student.data

                with torch.no_grad():
                    adapted_rppg = self.rppg_estimator_stu({'input_clip': clip_input})['rPPG']
                    hr_after = hr_bpm_from_rppg_batch(adapted_rppg)

                after_psds = []
                for batch_idx in range(adapted_rppg.shape[0]):
                    after_psd = cal_psd_hr(adapted_rppg[batch_idx], self.frame_rate, return_type='psd')
                    after_psds.append(after_psd)
                after_psds = torch.stack(after_psds, 0)

                gt_flat = clip_average_HR.detach().view(-1).cpu()
                before_flat = hr_before.detach().view(-1).cpu()
                pseudo_flat = pseudo_hr.detach().view(-1).cpu()
                after_flat = hr_after.detach().view(-1).cpu()

                debug_slots = min(before_flat.numel(), pseudo_flat.numel(), after_flat.numel())

                self.logger.info(
                    f'[STEPDBG] dataset_idx={dataset_idx} clip_group={clip_idx} '
                    f'fre_loss={fre_loss.item():.4f} kl_loss={kl_loss.item():.4f} '
                    f'selected_after_priors={selected_after_priors} selected_after_rs={selected_after_rs} '
                    f'changed_param_count={changed_param_count} param_delta_sum={param_delta_sum:.8f} param_delta_max={param_delta_max:.8f}'
                )

                for batch_idx in range(debug_slots):
                    if debug_seen >= debug_max_clips:
                        break

                    gt_idx = batch_idx if (gt_flat.numel() > 1 and batch_idx < gt_flat.numel()) else 0

                    gt_bpm = float(gt_flat[gt_idx].item())
                    before_bpm = float(before_flat[batch_idx].item())
                    pseudo_bpm = float(pseudo_flat[batch_idx].item())
                    after_bpm = float(after_flat[batch_idx].item())

                    err_before = abs(before_bpm - gt_bpm)
                    err_after = abs(after_bpm - gt_bpm)

                    rppg_l2 = torch.norm(adapted_rppg[batch_idx] - origional_rppg[batch_idx], p=2).item()
                    psd_l2 = torch.norm(after_psds[batch_idx] - origional_psds[batch_idx], p=2).item()

                    self.logger.info(
                        f'[ADAPTDBG] dataset_idx={dataset_idx} clip_group={clip_idx} batch_idx={batch_idx} '
                        f'gt={gt_bpm:.2f} before={before_bpm:.2f} pseudo={pseudo_bpm:.2f} after={after_bpm:.2f} '
                        f'err_before={err_before:.2f} err_after={err_after:.2f} delta_err={err_after - err_before:.2f} '
                        f'rppg_l2={rppg_l2:.8f} psd_l2={psd_l2:.8f} '
                        f'fre_loss={fre_loss.item():.4f} kl_loss={kl_loss.item():.4f} '
                        f'selected_after_priors={selected_after_priors} selected_after_rs={selected_after_rs}'
                    )
                    debug_seen += 1

            ## Final inference after adapting all clips of the video
            torch.save(
                self.rppg_estimator_stu.state_dict(),
                os.path.join(self.save_ckpt_path, f'rppg_estimator_stu_dataset_{dataset_idx}_tmp.pth')
            )

            with torch.no_grad():
                num_clip = 3
                input_len = inputs.shape[2]
                input_len = input_len - input_len % (num_clip * 4)
                clip_len = input_len // num_clip
                inputs = inputs[:, :, :input_len, :, :]
                ecg = ecg[:, :input_len]

                new_args = deepcopy(self.args)
                new_args.num_rppg = clip_len
                val_rppg_estimator = build_model(new_args).to(self.device)
                val_rppg_estimator.load_state_dict(
                    torch.load(
                        f'{self.save_ckpt_path}/rppg_estimator_stu_dataset_{dataset_idx}_tmp.pth',
                        map_location=self.device
                    )
                )
                val_rppg_estimator.eval()

                psd_gt_total = 0
                psd_pred_total = 0
                for idx in range(num_clip):
                    inputs_iter = inputs[:, :, idx * clip_len:(idx + 1) * clip_len, :, :]
                    ecg_iter = ecg[:, idx * clip_len:(idx + 1) * clip_len]

                    psd_gt = cal_psd_hr(ecg_iter, self.frame_rate, return_type='psd')
                    psd_gt_total += psd_gt.view(-1).max(0)[1].cpu() + 40

                    all_inputs = {
                        'input_clip': inputs_iter,
                    }
                    outputs = val_rppg_estimator(all_inputs)
                    rPPG = outputs['rPPG']

                    psd_pred = cal_psd_hr(rPPG[0], self.frame_rate, return_type='psd')
                    psd_pred_total += psd_pred.view(-1).max(0)[1].cpu() + 40

                hr_pred.append(float((psd_pred_total / num_clip).item()))

                if self.args.eval_gt_mode == 'label':
                    # Use the stored HDF5 gt_hr / clip_avg_hr label as evaluation GT.
                    # This is also the HR target used by ce_loss during training.
                    hr_gt.append(float(clip_average_HR.detach().view(-1).float().mean().cpu().item()))
                else:
                    # Original diagnostic mode: recompute GT HR from ECG PSD.
                    hr_gt.append(float((psd_gt_total / num_clip).item()))

        cur_mae, cur_rmse, cur_sd, cur_r = self.update_best(-1, hr_pred, hr_gt, val_type='clip')

        online_csv = os.path.join(self.save_path, f'online_ctta_predictions_dataset_{dataset_idx}.csv')

        def _to_float(x):
            if hasattr(x, 'detach'):
                x = x.detach().cpu()
            if hasattr(x, 'item'):
                return float(x.item())
            return float(x)

        with open(online_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'gt_bpm', 'pred_bpm', 'abs_error'])
            for pred_idx, (gt_val, pred_val) in enumerate(zip(hr_gt, hr_pred)):
                gt_float = _to_float(gt_val)
                pred_float = _to_float(pred_val)
                writer.writerow([pred_idx, gt_float, pred_float, abs(pred_float - gt_float)])

        self.logger.info(f'online_ctta_predictions_csv: {online_csv}')

        torch.save(
            self.rppg_estimator_stu.state_dict(),
            os.path.join(self.save_ckpt_path, f'rppg_estimator_stu_dataset_{dataset_idx}.pth')
        )
        torch.save(
            self.rppg_estimator_teacher.state_dict(),
            os.path.join(self.save_ckpt_path, f'rppg_estimator_teacher_dataset_{dataset_idx}.pth')
        )

        return cur_mae, cur_rmse, cur_sd, cur_r

    def train(self, start_dataset_idx, continue_log=''):
        self.prepare_train(start_dataset_idx, continue_log)
        self.logger.info(
            f'================================== Current Log Time : {self.current_date} ================================== \n'
            f'prepare train, load ckpt and block gradient, start_dataset_idx: {start_dataset_idx}, gpu: {self.gpu_list}.\n'
            f'dataset: {self.args.datasets}, num_rppg: {self.args.num_rppg}, model: {self.args.model}, loss: {self.loss_funcs_weight}.\n'
            f'batch_size: {self.actual_batch_size}, lr: {self.args.lr}, optim: {self.args.optim}, scheduler: {self.args.scheduler}.\n'
            f'ablation_mode: {self.args.ablation_mode}\n'
            f'tta_aug_mode: {self.args.tta_aug_mode}\n'
            f'eval_gt_mode: {self.args.eval_gt_mode}'
        )

        if start_dataset_idx == 0:
            self.logger.info(f'===== Training at the dataset : {self.all_datasets[0]} =====')
            self.initial_train(dataset_idx=0)

        if self.args.source_only_eval:
            self.logger.info(f'===== SOURCE-ONLY EVAL at the dataset : {self.all_datasets[0]} =====')
            cur_mae, cur_rmse, cur_sd, cur_r = self.evaluate_clip(
                epoch=self.args.epochs - 1,
                val_dataloader=self.val_dataloaders[0]
            )
            self.logger.info(
                f'===== SOURCE-ONLY RESULT =====\n'
                f'MAE: {cur_mae}, RMSE: {cur_rmse}, SD: {cur_sd}, R: {cur_r}'
            )
            return

        mean_mae, mean_rmse, mean_sd, mean_r = [], [], [], []
        for i in range(max(1, start_dataset_idx), len(self.all_datasets)):
            self.logger.info(f'===== TTA at the dataset : {self.all_datasets[i]} =====')
            cur_mae, cur_rmse, cur_sd, cur_r = self.continue_tta(dataset_idx=i)
            mean_mae.append(cur_mae)
            mean_rmse.append(cur_rmse)
            mean_sd.append(cur_sd)
            mean_r.append(cur_r)

        self.logger.info(
            f'===== MEAN RESULTs at all datasets =====\n'
            f'MAE: {np.mean(mean_mae)}, RMSE: {np.mean(mean_rmse)}, SD: {np.mean(mean_sd)}, R: {np.mean(mean_r)}'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## general params.
    parser.add_argument('--num_rppg', type=int, default=160, help='the number of rPPG')
    parser.add_argument('--datasets', type=str, default='VIPL_UBFC_UBFCA_PURE_PUREA_BUAA_BUAAA', help='dataset')
    parser.add_argument('--vipl_fold', type=int, default=-1, help='the fold of VIPL dataset, not used')
    parser.add_argument('--save_path', type=str, default='path/to/your/save_dir', help='the path to save the model [ckpt, code, visulization]')
    parser.add_argument('--save_mode', type=str, default='all', help='save mode [all, best]')

    ## train params.
    parser.add_argument('--gpu', type=str, default="2", help='gpu id list')
    parser.add_argument('--img_size', type=int, default=128, help='the length of clip')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size per gpu')
    parser.add_argument('--eval_step', type=int, default=1, help='the number of **epochs** to eval')
    parser.add_argument('--epochs', type=int, default=20, help='the number of epochs to train')
    parser.add_argument('--echo_batches', type=int, default=500, help='the number of **mini-batches** to print the loss')

    ## loss
    parser.add_argument('--loss', type=str, default='["np_loss", "ce_loss"]', help='loss = [np_loss, ce_loss]')
    parser.add_argument('--loss_weight', type=str, default='[0, 1]', help='loss_weight = [1, 1]')

    ## model params.
    parser.add_argument('--model', type=str, default='ResNet3D', help='model')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')

    ## optim
    parser.add_argument('--optim', type=str, default='adam', help='optimizer = [adam, sgd]')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')

    ## scheduler
    parser.add_argument('--scheduler', type=str, default='step', help='scheduler = [step]')
    parser.add_argument('--step_size', type=int, default=50, help='learning rate decay step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay')

    ## Table 2 / TTA hyperparameters
    parser.add_argument('--tta_num_augs', type=int, default=10)
    parser.add_argument('--tta_future_steps', type=int, default=4)
    parser.add_argument('--tta_save_ratio', type=float, default=0.8)
    parser.add_argument('--tta_related_save_ratio', type=float, default=0.2)
    parser.add_argument('--tta_teacher_alpha', type=float, default=0.99)

    ## Table 3 / ablation
    parser.add_argument(
        '--ablation_mode',
        type=str,
        default='full',
        choices=['full', 'priors_only', 'priors_pa', 'priors_rs']
    )

    parser.add_argument(
        '--tta_aug_mode',
        type=str,
        default='all',
        choices=[
            'all',
            'identity',
            'gaussian_only',
            'crop_only',
            'flip_only',
            'reverse_only',
            'gaussian_crop'
        ]
    )

    parser.add_argument('--source_only_eval', action='store_true')

    parser.add_argument(
        '--eval_gt_mode',
        type=str,
        default='label',
        choices=['label', 'ecg_psd'],
        help='GT source for evaluation: stored HDF5 gt_hr/clip_avg_hr label or ECG-PSD-derived HR'
    )

    args = parser.parse_args()

    set_seed(92)

    rppg_estimator_trainer = RppgEstimatorTrainer(args)
    rppg_estimator_trainer.train(start_dataset_idx=0, continue_log='')