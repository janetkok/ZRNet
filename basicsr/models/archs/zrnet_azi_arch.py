import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.init as init
from basicsr.models.archs.kb_utils import KBAFunction
from basicsr.models.archs.kb_utils import LayerNorm2d, SimpleGate
from basicsr.models.relation import GraphAttentionLayer


class Mlp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Mlp, self).__init__()
        self.layers = nn.Sequential( 
            nn.Linear(in_ch, in_ch*2),
            nn.GELU(),
            nn.Linear(in_ch*2, out_ch),
            nn.GELU(),
        )
        self.layers[0].apply(self.initialize_weights)
        self.layers[2].apply(self.initialize_weights)

    def initialize_weights(self,module):

        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(module.bias, -bound, bound)

    def forward(self, x):
        return self.layers(x)
    

class MlpPred(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MlpPred, self).__init__()
        self.layers = nn.Sequential( 
            nn.Linear(in_ch, in_ch*2),
            nn.GELU(),
            nn.Linear(in_ch*2, out_ch)
        )
        self.layers[0].apply(self.initialize_weights)
        self.layers[2].apply(self.initialize_weights)

    def initialize_weights(self,module):

        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(module.bias, -bound, bound)

    def forward(self, x):
        return self.layers(x)

class KBBlock_s(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False):
        super(KBBlock_s, self).__init__()
        self.k, self.c = k, c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,
                          bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                          bias=True),
            )

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                                bias=True)

        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )

        self.conv211 = nn.Conv2d(in_channels=c, out_channels=self.nset, kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=dw_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_ch, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.sg = SimpleGate()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        sca = self.sca(x)
        x1 = self.conv11(x)

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
        x = x * x1 * sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        
        # FFN
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma


class ZRNet_azi(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8],
                 dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s', lightweight=False, ffn_scale=2,num_zernike = 25,middle_blk_zern_num= 1):
        super().__init__()
        out_channel = 1
        self.num_zernike = num_zernike
        basicblock = eval(basicblock)

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )

            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(middle_blk_num)]
            )
        
        self.middle_blks_zern = \
            nn.Sequential(
                *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(middle_blk_zern_num)]
            )
        self.zern_flat = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )

        self.zern_dim = 256
        self.zern_mlp = nn.ModuleList()
        self.zern_mlp_pred = nn.ModuleList()
        
        for i in range(self.num_zernike):
            self.zern_mlp.add_module('zern_mlp_{}'.format(i), Mlp(chan,self.zern_dim))
            self.zern_mlp_pred.add_module('zern_mlp_pred{}'.format(i), MlpPred(self.zern_dim,1))

        self.relation_zern_s2_n2n = nn.ModuleList()
        self.relation_zern_s2_n2r = nn.ModuleList()
        self.relation_zern_s2_r2n = nn.ModuleList()

        self.relation_zern_s3_n2n = nn.ModuleList()
        self.relation_zern_s3_n2r = nn.ModuleList()
        self.relation_zern_s3_r2n = nn.ModuleList()

        self.n_s1 = 4
        self.n_s2 = 6
        self.n_s3 = 3
        self.total_ab_g = self.n_s1 + self.n_s2 + self.n_s3 

        for i in range(self.n_s2):
            self.relation_zern_s2_n2n.add_module('zern_s2_subg_n2n_{}'.format(i), GraphAttentionLayer(self.zern_dim, self.zern_dim, 0.2, 0.2, concat=True))
            self.relation_zern_s2_n2r.add_module('zern_s2_subg_n2r_{}'.format(i), GraphAttentionLayer(self.zern_dim, self.zern_dim, 0.2, 0.2, concat=True))
            self.relation_zern_s2_r2n.add_module('zern_s2_subg_r2n_{}'.format(i), GraphAttentionLayer(self.zern_dim, self.zern_dim, 0.2, 0.2, concat=True))
        for i in range(self.n_s3):
            self.relation_zern_s3_n2n.add_module('zern_s3_subg_n2n_{}'.format(i), GraphAttentionLayer(self.zern_dim, self.zern_dim, 0.2, 0.2, concat=True))
            self.relation_zern_s3_n2r.add_module('zern_s3_subg_n2r_{}'.format(i), GraphAttentionLayer(self.zern_dim, self.zern_dim, 0.2, 0.2, concat=True))
            self.relation_zern_s3_r2n.add_module('zern_s3_subg_r2n_{}'.format(i), GraphAttentionLayer(self.zern_dim, self.zern_dim, 0.2, 0.2, concat=True))

        self.relation_zern_r2r = GraphAttentionLayer(self.zern_dim, self.zern_dim, 0.2, 0.2, concat=True)
  

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )


        self.padder_size = 2 ** len(self.encoders)


    def forward(self, inp):
        batch = inp.shape[0]
        x = self.intro(inp)

        encs = []

        for encoder,  down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        zernike2d = self.middle_blks_zern(x)
        zernike1d = self.zern_flat(zernike2d)

        s1_unit = torch.zeros(self.n_s1, batch,1,self.zern_dim, device='cuda')
        s2_unit = torch.zeros(self.n_s2, batch,2,self.zern_dim, device='cuda') 
        s3_unit = torch.zeros(self.n_s3, batch,3,self.zern_dim, device='cuda') 
        s2_r = torch.zeros(self.n_s2,batch,1,self.zern_dim, device='cuda') 
        s3_r = torch.zeros(self.n_s3,batch,1,self.zern_dim, device='cuda') 

        #separate MLPs
        for i,zi in zip(range(self.n_s1),[12,17,18,24]):
            s1_unit[i,:,0,:] =  getattr(self.zern_mlp, f'zern_mlp_{zi}')(zernike1d)
        for i,zi in zip(range(self.n_s2),[[3,13],[4,14],[5,15],[6,16],[7,19],[11,23]]): 
            for j in range(2):
                s2_unit[i,:,j,:] = getattr(self.zern_mlp, f'zern_mlp_{zi[j]}')(zernike1d)
        for i,zi in zip(range(self.n_s3),[[0,8,20],[1,9,21],[2,10,22]]): 
            for j in range(3):
                s3_unit[i,:,j,:] = getattr(self.zern_mlp, f'zern_mlp_{zi[j]}')(zernike1d)

        # initialise Zernike nodes and group nodes
        for i in range(self.n_s2):
            s2_u_temp = s2_unit[i,:,:,:].clone()
            s2_r[i,:,:,:] = torch.mean(s2_u_temp, dim=1).unsqueeze(1)
        for i in range(self.n_s3):
            s3_u_temp = s3_unit[i,:,:,:].clone()
            s3_r[i,:,:,:] = torch.mean(s3_u_temp, dim=1).unsqueeze(1) 

        adj_s2_n2n = torch.ones(2,2, device='cuda')
        adj_s2_n2r = torch.eye(3, device='cuda')
        adj_s2_n2r[2,:] = 1

        adj_s3_n2n = torch.ones(3,3, device='cuda')
        adj_s3_n2r = torch.eye(4, device='cuda')
        adj_s3_n2r[3,:] = 1

        adj_r2r = torch.ones(self.total_ab_g,self.total_ab_g, device='cuda')

        adj_s2_r2n = torch.eye(3, device='cuda')
        adj_s2_r2n[:,2] = 1

        adj_s3_r2n = torch.eye(4, device='cuda') 
        adj_s3_r2n[:,3] = 1


        for b in range(batch): 
            # information exchange and aggregation
            for ab_g in range(self.n_s2): 
                s2_unit_data = s2_unit[ab_g,b,:,:].clone()
                s2_unit[ab_g,b,:,:] = getattr(self.relation_zern_s2_n2n, 'zern_s2_subg_n2n_{}'.format(ab_g))(s2_unit_data, adj_s2_n2n)
                s2_u_r = torch.cat([s2_unit[ab_g,b,:,:], s2_r[ab_g,b,:,:]], dim=0) 
                res_s2_n2r = getattr(self.relation_zern_s2_n2r, 'zern_s2_subg_n2r_{}'.format(ab_g))(s2_u_r, adj_s2_n2r) 
                s2_r[ab_g,b,:,:]= res_s2_n2r[2,:] 
            for ab_g in range(self.n_s3): 
                s3_unit_data = s3_unit[ab_g,b,:,:].clone()
                s3_unit[ab_g,b,:,:] = getattr(self.relation_zern_s3_n2n, 'zern_s3_subg_n2n_{}'.format(ab_g))(s3_unit_data, adj_s3_n2n)
                s3_u_r = torch.cat([s3_unit[ab_g,b,:,:], s3_r[ab_g,b,:,:]], dim=0) 
                res_s3_n2r = getattr(self.relation_zern_s3_n2r, 'zern_s3_subg_n2r_{}'.format(ab_g))(s3_u_r, adj_s3_n2r) 
                s3_r[ab_g,b,:,:]= res_s3_n2r[3,:] 

            # information exchange between azimtuhal degree groups
            r_data = torch.cat([s1_unit[:,b,:,:], s2_r[:,b,:,:],s3_r[:,b,:,:]],dim=0).squeeze() 
            r = self.relation_zern_r2r(r_data,adj_r2r)

            s1_unit[:,b,0,:] = r[:self.n_s1,:]
            s2_r[:,b,0,:] = r[self.n_s1:self.n_s1+self.n_s2,:]
            s3_r[:,b,0,:] = r[self.n_s1+self.n_s2:,:]

            # information feedback
            for ab_g in range(self.n_s2):  
                s2_u_r = torch.cat([s2_unit[ab_g,b,:,:], s2_r[ab_g,b,:,:]], dim=0) 
                res_s2_r2n = getattr(self.relation_zern_s2_r2n, 'zern_s2_subg_r2n_{}'.format(ab_g))(s2_u_r, adj_s2_r2n)
                s2_unit[ab_g,b,:,:] = res_s2_r2n[:2,:] 
            for ab_g in range(self.n_s3):  
                s3_u_r = torch.cat([s3_unit[ab_g,b,:,:], s3_r[ab_g,b,:,:]], dim=0) 
                res_s3_r2n = getattr(self.relation_zern_s3_r2n, 'zern_s3_subg_r2n_{}'.format(ab_g))(s3_u_r, adj_s3_r2n)
                s3_unit[ab_g,b,:,:] = res_s3_r2n[:3,:] 

        # separate MLPs
        zernike = torch.zeros(batch,self.num_zernike, device='cuda')
        for i,zi in zip(range(self.n_s1),[12,17,18,24]):
            zernike[:,zi]=getattr(self.zern_mlp_pred, f'zern_mlp_pred{zi}')(s1_unit[i,:,0,:]).squeeze()
        for i,zi in zip(range(self.n_s2),[[3,13],[4,14],[5,15],[6,16],[7,19],[11,23]]):
            for j in range(2):
                zernike[:,zi[j]]=getattr(self.zern_mlp_pred, f'zern_mlp_pred{zi[j]}')(s2_unit[i,:,j,:]).squeeze()
        for i,zi in zip(range(self.n_s3),[[0,8,20],[1,9,21],[2,10,22]]):
            for j in range(3):
                zernike[:,zi[j]]=getattr(self.zern_mlp_pred, f'zern_mlp_pred{zi[j]}')(s3_unit[i,:,j,:]).squeeze()

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp[:,1,:,:].unsqueeze(1)

        return x, zernike

