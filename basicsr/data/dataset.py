
from torch.utils import data
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import torch
import math

import torch.nn.functional as F
import numpy as np
import pandas as pd

import os


def q_sample(x_start, aberration):
    # batch convolve
    psf=aberration*aberration
    psf_div = psf.sum(dim=(2,3))
    psf=psf/(psf_div[:,:,None,None])# Normalize area to 1

    Fpsf=torch.fft.fft2(torch.fft.fftshift(psf,dim=(2,3)),dim=(2,3))
    Fpicy=torch.fft.fft2(torch.fft.fftshift(x_start,dim=(2,3)),dim=(2,3))
    outfft=torch.fft.ifftshift(torch.fft.ifft2(Fpicy*Fpsf,dim=(2,3)),dim=(2,3))
    outfft=abs(outfft)

    return outfft

def q_sample_fft(x_start, aberration):
    # batch convolve
    psf=aberration*aberration
    psf_div = psf.sum(dim=(2,3))
    psf=psf/(psf_div[:,:,None,None])# Normalize area to 1

    Fpsf=torch.fft.fft2(torch.fft.fftshift(psf,dim=(2,3)),dim=(2,3))
    Fpicy=torch.fft.fft2(torch.fft.fftshift(x_start,dim=(2,3)),dim=(2,3))

    return Fpicy*Fpsf


class Aberration_CONV3():
    def __init__(self,img_size,device='cpu',precision=torch.float,zernike=[4,5,6,7,8,9],zRange=1.0):
        self.zRange_start = 0
        self.zRange_end = 200 #0-200 ->-1.0 to 1.0
        if zRange==0.5:
            self.zRange_start = 50
            self.zRange_end = 150 
        self.zRange = zRange
        self.tRange = int(zRange*100)

        self.img_size = img_size
        self.device = device
        self.zernike = zernike
        self.num_zernike = len(zernike)
        self.precisionFloat = precision
        self.precisionInt = torch.int

        self.nrange =int(self.img_size/2)
        self.npts=int(np.floor(64/self.nrange*491)) 
        self.npad=2001 
        self.nhpad=math.ceil(self.npad/2)-1
        self.nex=int(round(self.npad-self.npts)/2)

        self.znorm=0
        self.phaseapp=self.znorm*math.pi/2
        self.z_sub = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6]
        self.defocus = [0.0]

        self.obs = [self.npts/self.npad,1,1]
        self.pup = self.annulus()
        self.circ_res = self.circ(self.npts)
        self.rad= self.circ_res*self.circrad(self.npts)
        self.rad=F.pad(self.rad,(self.nex,self.nex,self.nex,self.nex))
        self.pupPhase = self.pup*torch.exp(1j*self.phaseapp*self.rad*self.rad)

        self.zernike_cartesian_matrix()


    def annulus(self):
        if self.obs[2]>1:
            print('error outer ring out of aperture')
            return

        mattemp = self.circrad(self.npad)

        pup=torch.zeros(self.npad,self.npad,device=self.device,dtype=self.precisionFloat)
        temp1=self.obs[0]*torch.ones(self.npad,self.npad,device=self.device,dtype=self.precisionFloat)
        temp2=self.obs[1]*torch.ones(self.npad,self.npad,device=self.device,dtype=self.precisionFloat)
        temp3=self.obs[2]*torch.ones(self.npad,self.npad,device=self.device,dtype=self.precisionFloat)
        pup=(mattemp<=temp1)|((mattemp>temp2)&(mattemp<=temp3))

        nh=math.ceil(self.npad/2)-1
        if self.obs[0]==0:
            pup[nh,nh]=0 # just blocks central pixel

        return pup

    def circrad(self,npts):

        xtemp = torch.linspace(-1,1,steps=npts,device=self.device,dtype=self.precisionFloat)
        ytemp=xtemp

        mattemp1 = xtemp.view(len(xtemp),1)@torch.ones(1,npts,device=self.device,dtype=self.precisionFloat)
        mattemp2 = 1j*torch.ones(npts,1,device=self.device)*ytemp
        mattemp = mattemp1+mattemp2
        mattemp = abs(mattemp)

        return mattemp

    def circ(self,npts):
        mattemp = self.circrad(npts)
        return (mattemp<=torch.ones(npts,npts,device=self.device))

    def zernike_cartesian_matrix(self):
        # my simplified version of zernike polynomial function 
        # Gives Zernikes in Cartesians over a unit circle.
        # param: subscript superscript azimuthal angle
        x=torch.linspace(-1,1,steps=self.npts,device=self.device,dtype=self.precisionFloat) #Unit radius
        x=torch.flip(x,dims=(0,))
        x = x.repeat(self.npts,1)
        y=x.t()
        #dx=2/(self.npts-1)
        Z = torch.zeros(28,self.npts,self.npts,dtype=self.precisionFloat,device=self.device)
        Z[0,:,:] = 1*torch.ones(self.npts,self.npts,dtype=self.precisionFloat,device=self.device)
        Z[1,:,:] = 2*x #tilt
        Z[2,:,:] = 2*y
        Z[3,:,:] = math.sqrt(6)*(2*x*y) #astigmatism
        Z[4,:,:] = math.sqrt(3)*((2*x*x)+(2*y*y)-1) #defocus
        Z[5,:,:] = math.sqrt(6)*((-x*x)+(y*y)) #astigmatism
        Z[6,:,:] = 2*math.sqrt(2)*((-x**3)+(3*x*y*y)) #trefoil
        Z[7,:,:] = 2*math.sqrt(2)*((-2*x)+(3*x**3)+(3*x*y*y))#coma
        Z[8,:,:] = 2*math.sqrt(2)*((-2*y)+(3*y**3)+(3*y*x*x)) #coma
        Z[9,:,:] = 2*math.sqrt(2)*((-y**3)+(3*y*x*x)) #trefoil
        Z[10,:,:] = math.sqrt(10)*((-4*x**3*y)+(4*x*y**3))
        Z[11,:,:] = math.sqrt(10)*((-6*x*y)+(8*y*x**3)+(8*x*y**3))
        Z[12,:,:] = math.sqrt(5)*(1-(6*x*x)-(6*y*y)+(6*x**4)+(12*x*x*y*y)+(6*y**4))
        Z[13,:,:] = math.sqrt(10)*((3*x*x)-(3*y*y)-(4*x**4)+(4*y**4))
        Z[14,:,:] = math.sqrt(10)*((x**4)-(6*y**2*x**2)+(y**4))
        Z[15,:,:] = 2*math.sqrt(3)*((x**5)-(10*x**3*y**2)+(5*x*y**4))
        Z[16,:,:] = 2*math.sqrt(3)*((4*x**3)-(12*y**2*x)-(5*x**5)+(10*y**2*x**3)+(15*y**4*x))
        Z[17,:,:] = 2*math.sqrt(3)*((3*x)-(12*x**3)-(12*x*y**2)+(10*x**5)+(20*x**3*y**2)+(10*x*y**4))
        Z[18,:,:] = 2*math.sqrt(3)*((3*y)-(12*y**3)-(12*y*x**2)+(10*y**5)+(20*x**2*y**3)+(10*y*x**4))
        Z[19,:,:] = 2*math.sqrt(3)*(-(4*y**3)+(12*x**2*y)+(5*y**5)-(10*x**2*y**3)-(15*x**4*y))
        Z[20,:,:] = 2*math.sqrt(3)*((y**5)-(10*y**3*x**2)+(5*y*x**4))
        Z[21,:,:] = math.sqrt(14)*((6*x**5*y)-(20*x**3*y**3)+(6*x*y**5))
        Z[22,:,:] = math.sqrt(14)*((20*x**3*y)-(20*x*y**3)-(24*x**5*y)+(24*x*y**5))
        Z[23,:,:] = math.sqrt(14)*((12*x*y)-(40*x**3*y)-(40*x*y**3)+(30*x**5*y)+(60*x**3*y**3)+(30*x*y**5))
        Z[24,:,:] = math.sqrt(7)*(-1+(12*x*x)+(12*y*y)-(30*x**4)-(60*x**2*y**2)-(30*y**4)+(20*x**6)+(60*x**4*y**2)+(60*x**2*y**4)+(20*y**6))
        Z[25,:,:] = math.sqrt(14)*(-(6*x**2)+(6*y**2)+(20*x**4)-(20*y**4)-(15*x**6)-(15*x**4*y**2)+(15*x**2*y**4)+(15*y**6))
        Z[26,:,:] = math.sqrt(14)*(-(5*x**4)+(30*x**2*y**2)-(5*y**4)+(6*x**6)-(30*x**4*y**2)-(30*y**4*x**2)+(6*y**6))
        Z[27,:,:] = math.sqrt(14)*(-(x**6)+(15*x**4*y**2)-(15*y**4*x**2)+(y**6))

        self.Z = Z*self.circ_res


    def gen(self,batch_size=32,C=None,og=None):
  
        convolved = torch.empty(size=(batch_size, 3,self.img_size,self.img_size),device=self.device)
        aberration = torch.empty(size=(batch_size, 3,self.img_size,self.img_size),device=self.device)

        for i in range(batch_size):


            Zsum_noDef=torch.zeros(self.npts,self.npts,device=self.device,dtype=self.precisionFloat)
            Zsum=torch.zeros(self.npts,self.npts,device=self.device,dtype=self.precisionFloat)
            for j,m1 in enumerate(self.zernike):
                if C[i,j]!=0:
                    Zsum_noDef=Zsum_noDef + (C[i,j]*self.Z[m1,:,:])

            for h,k in zip([1,0,2],[0,-1,1]):    
                if k!=0:
                    Zsum = Zsum_noDef + (k*self.Z[4,:,:])
                else:
                    Zsum = Zsum_noDef 

                Zpadsum=F.pad(Zsum,(self.nex,self.nex,self.nex,self.nex)) 
                del Zsum
                Zpadsum=self.pupPhase*torch.exp(1j*Zpadsum) #includes phase from physical defocus

                Zpadsum=torch.fft.ifftshift(Zpadsum)
                out=torch.fft.fft2(Zpadsum)
                del Zpadsum
                out=torch.fft.fftshift(out)
                out=abs(out)
                
                outsmall=out[self.nhpad-(self.nrange-1):self.nhpad+(self.nrange+1),self.nhpad-(self.nrange-1):self.nhpad+(self.nrange+1)]
                outsmall=outsmall/(torch.max(outsmall))
                aberration[i,h,:,:] = outsmall.detach().clone()
                convolved[i,h,:,:]=q_sample(og[i,0,:,:].unsqueeze(0).unsqueeze(0),outsmall.unsqueeze(0).unsqueeze(0))

        return convolved, C, aberration




class CytoImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, image_size, device='cpu',precision=torch.float,zernike=[3,4,5,6,7],zRange=1.0, split='train',exts = ['jpg', 'jpeg', 'png','PNG']):
        
        
        # # Use metadata to create generators of image batches
        self.split=split
        df = pd.read_csv(data_path)
        root_db = data_path.split('/cytoimagenet')[0]
        self.df = df[df.dset.isin([split])]
        self.df  = self.df.reset_index(drop=True)

        self.zRange_start = 0
        self.zRange_end = 200 #0-200 ->-1.0 to 1.0
        self.zRange = zRange
        if zRange==0.5:
            self.zRange_start = 50
            self.zRange_end = 150 
        self.tRange = int(zRange*100)

        self.num_zernike = len(zernike)
        self.C_val_test = None
        if split=='val':
            C_val_test_df_path = root_db + '/'+str(self.zRange_start)+'_'+str(self.zRange_end)+'_'+str(self.num_zernike)+'.csv'
            if os.path.isfile(C_val_test_df_path): #load test csv
                C_val_test_df = pd.read_csv(C_val_test_df_path)
                self.C_val_test = torch.tensor(C_val_test_df.values).float()
            else: #generate test csv
                self.C_val_test = torch.randint(self.zRange_start, self.zRange_end ,size=(len(self.df),self.num_zernike))
                self.C_val_test = (self.C_val_test-100)*0.01
                pd.DataFrame(self.C_val_test.numpy()).to_csv(C_val_test_df_path,index=False)
        self.device=device
        self.gen_aberration = Aberration_CONV3(image_size, device,precision,zernike,zRange)
        self.image_size = image_size

        self.transform = transforms.Compose([
                transforms.Resize((image_size,image_size)),
            ])

    def __getitem__(self,idx):
        path = self.df.full_path[idx]
        img = self.transform(Image.open(path))
        img = torch.from_numpy(np.asarray(img)/255).float().unsqueeze(0)
        if self.split=='val':
            convolved,coeffs, aberration = self.gen_aberration.gen(batch_size=1,C=self.C_val_test[idx].unsqueeze(0),og=img.unsqueeze(0))
        else:
            C = torch.randint(self.zRange_start, self.zRange_end ,size=(1,self.num_zernike))
            C = (C-100)*0.01

            convolved,coeffs, aberration = self.gen_aberration.gen(batch_size=1,C=C,og=img.unsqueeze(0))
            
        convolved = convolved.squeeze()
        coeffs = coeffs.squeeze()
        aberration = aberration.squeeze()
        return convolved,img,aberration,coeffs

    def __len__(self):
        return len(self.df)


    
class Conv3_fft(Aberration_CONV3):
    def __init__(self,img_size,,device='cpu',precision=torch.float,zernike=[4,5,6,7,8,9],zRange=1.0):
        super().__init__(img_size,device=device,precision=precision,zernike=zernike,zRange=zRange)


    def gen(self,batch_size=32,C=None,og=None):
        convolved_real = torch.empty(size=(batch_size, 1,self.img_size,self.img_size),device=self.device)
        convolved_imag = torch.empty(size=(batch_size, 1,self.img_size,self.img_size),device=self.device)
        aberration = torch.empty(size=(batch_size, 1,self.img_size,self.img_size),device=self.device)

        for i in range(batch_size):

            Zsum=torch.zeros(self.npts,self.npts,device=self.device,dtype=self.precisionFloat)
            for j,m1 in enumerate(self.zernike): 
                if C[i,j]!=0:
                    Zsum=Zsum + (C[i,j]*self.Z[m1,:,:])

            Zpadsum=F.pad(Zsum,(self.nex,self.nex,self.nex,self.nex)) 
            del Zsum
            Zpadsum=self.pupPhase*torch.exp(1j*Zpadsum) #includes phase from physical defocus

            Zpadsum=torch.fft.ifftshift(Zpadsum)
            out=torch.fft.fft2(Zpadsum)
            del Zpadsum
            out=torch.fft.fftshift(out)
            out=abs(out)

            outsmall=out[self.nhpad-(self.nrange-1):self.nhpad+(self.nrange+1),self.nhpad-(self.nrange-1):self.nhpad+(self.nrange+1)]
            outsmall=outsmall/(torch.max(outsmall))
            aberration[i,0,:,:] = outsmall.detach().clone()
            convolved=q_sample_fft(og[i,0,:,:].unsqueeze(0).unsqueeze(0),outsmall.unsqueeze(0).unsqueeze(0))
            convolved_real[i,0,:,:] = convolved.real
            convolved_imag[i,0,:,:] = convolved.imag

        return convolved_real, convolved_imag




