import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img, psnr, ssim
loss_module = importlib.import_module('basicsr.models.losses')

import os

import torch.nn.functional as F
from torchvision import utils

import lpips
from basicsr.data.dataset import Conv3_fft
from pathlib import Path



class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
        self.num_zernike = self.opt['network_g'].pop('num_zernike')
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')
        if train_opt.get('zern_opt'):
            zern_type = train_opt['zern_opt'].pop('type')
            cri_zern_cls = getattr(loss_module, zern_type)
            self.cri_zern = cri_zern_cls(**train_opt['zern_opt']).to(
                self.device)
        if train_opt.get('faa_opt'):
            abb_type = train_opt['faa_opt'].pop('type')
            cri_abb_cls = getattr(loss_module, abb_type)
            self.cri_abb = cri_abb_cls(**train_opt['faa_opt']).to(
                self.device)
            self.gen_aberration = Conv3_fft(img_size=self.opt['image_size'][0],device='cuda',zernike=self.opt['zernike'],zRange=self.opt['z_range'])

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.zern_gt = data['zern_gt'].to(self.device)
        

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.zern_gt = data['zern_gt'].to(self.device)

    def feed_data_nonpad_test(self,data):     
        
        lq = data['lq']

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred,zernike = self.net_g_ema(lq)
            if isinstance(pred, list):
                pred = pred[-1]
            output = pred
            output_z = zernike
            return output,output_z
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred,zernike = self.net_g(lq)
            if isinstance(pred, list):
                pred = pred[-1]
            output = pred
            output_z = zernike
            self.net_g.train()
            return output,output_z


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds, zernike = self.net_g(self.lq)

        if not isinstance(preds, list):
            preds = [preds]
        
        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        l_zern = 0.
        l_faa = torch.tensor(0.,device='cuda')

        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)
            l_zern += self.cri_zern(zernike, self.zern_gt)

            if self.opt['train'].get('faa_opt'):
                # compute FFT of the convolution between the image and aberration
                res_zgt_real, res_zgt_imag = self.gen_aberration.gen(batch_size=pred.size()[0],C=self.zern_gt,og=pred)
                og_zpred_real, og_zpred_imag = self.gen_aberration.gen(batch_size=pred.size()[0],C=zernike,og=self.gt)
                og_zgt_real, og_zgt_imag = self.gen_aberration.gen(batch_size=pred.size()[0],C=self.zern_gt,og=self.gt)
                # restoration loss
                l_faa += self.cri_abb(res_zgt_real, og_zgt_real)
                l_faa += self.cri_abb(res_zgt_imag, og_zgt_imag)
                # Zernike loss
                l_faa += self.cri_abb(og_zpred_real, og_zgt_real)
                l_faa += self.cri_abb(og_zpred_imag, og_zgt_imag)
                # cross-verification loss
                l_faa += self.cri_abb(res_zgt_real, og_zpred_real)
                l_faa += self.cri_abb(res_zgt_imag, og_zpred_imag)
            
        loss_dict['l_pix'] = l_pix
        loss_dict['l_zern'] = l_zern
        loss_dict['l_faa'] = l_faa

        loss_comb = l_pix + l_zern + l_faa
        loss_comb.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred,zernike = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.output_z = zernike
            return self.output,self.output_z
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred,zernike = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.output_z = zernike
            self.net_g.train()
            return self.output,self.output_z

    def infer(self, dataloader, current_iter, tb_logger,
                           save_img,num_val):
        mse_loss = torch.nn.MSELoss()
        lpips_vgg = lpips.LPIPS(net='vgg').cuda()
        if os.environ['LOCAL_RANK'] == '0':
            save_folder = Path(osp.join(self.opt['path']['visualization'],
                                                f'{num_val}_{current_iter}'))
            save_folder.mkdir(exist_ok = True)

            self.metric_results = {
                    'psnr': 0,
                    'ssim': 0,
                    'lpips':0,
                    'zernike_mse':0,
                    'zernike_rmswfe':torch.zeros(self.num_zernike).to(self.device)
                }

            test = self.nonpad_test
            if len(dataloader)<=num_val:
                num_val = len(dataloader)
            for idx,(lq,gt,_,zern_gt) in enumerate(dataloader):
                if idx==(num_val):
                    break 
                lq, gt = lq.cuda(), gt.cuda()
                self.feed_data({'lq': lq, 'gt':gt, 'zern_gt': zern_gt})
                test()
                og_img = torch.clamp(gt,min=0,max=1)
                recons = torch.clamp(self.output,min=0,max=1)
                self.metric_results['psnr'] += psnr(og_img, recons, data_range=(0,1)).item()
                self.metric_results['ssim'] +=  ssim(og_img, recons, data_range=(0,1)).item()
                self.metric_results['lpips'] +=  lpips_vgg(og_img, recons,normalize=True).item()
                self.metric_results['zernike_mse'] += mse_loss(self.output_z, self.zern_gt)
                self.metric_results['zernike_rmswfe'] += ((self.output_z - self.zern_gt) ** 2).sum(0)
                

                if save_img:
                    print(idx)
                    print('output')
                    print(self.output_z)
                    print('ground truth')
                    print(self.zern_gt)
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                                f'{num_val}_{current_iter}',
                                                f'sample-recon-test_{idx}.png')
                    
                    # save_gt_img_path = osp.join(self.opt['path']['visualization'],
                    #                             f'{num_val}_{current_iter}',
                    #                             f'sample-og-test_{idx}.png')
                                                
                    # utils.save_image(og_img, save_gt_img_path, nrow=1)
                    utils.save_image(recons, save_img_path, nrow = 1)
                    # utils.save_image(lq[0,:,:,:], osp.join(self.opt['path']['visualization'],
                    #                         f'{num_val}_{current_iter}',
                    #                         f'sample-xt-test_{idx}.png'))

                    # for j in range(3):
                    #     temp_xt = lq[:,j,:,:]
                    #     temp_xt = temp_xt[:,None,:,:]
                    #     utils.save_image(temp_xt, osp.join(self.opt['path']['visualization'],
                    #                             f'{num_val}_{current_iter}',
                    #                             f'sample-xt-test_{idx}_{j}.png'),
                    #                         nrow=1)       
                # tentative for out of GPU memory
                del self.lq
                del self.output
                del self.output_z
                del self.gt
                del self.zern_gt
                torch.cuda.empty_cache()
                
            for metric in self.metric_results.keys():
                if metric=='zernike_rmswfe':
                    self.metric_results['zernike_rmswfe']= round(torch.sqrt((self.metric_results['zernike_rmswfe'] / num_val).sum()).item(), 4)
                else:
                    self.metric_results[metric] /= num_val


            self._log_validation_metric_values(current_iter, self.opt['dataset_name'],
                                                tb_logger)
            return 
        else:
            return 0

   
    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)



    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

