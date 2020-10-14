# Created on 2020/10
# Author: Yimingxiao

import argparse
import os

import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
from data import AudioDataLoader, AudioDataset
from pit_criterion import cal_loss
from conv_tasnet_tfse1 import ConvTasNet
from utils import remove_pad


parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
parser.add_argument('--model_path', type=str, default='ClustertrainTFSE1New/final_paper_2_3_2cho_continue.pth.tar'
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=str, default='../tools/valDataNewTwo/',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--cal_sdr', type=int, default=1,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')


def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0
    avg_SISNRiPitNum = 0
    length = torch.ones(1)
    length = length.int()
    numberEsti =[]
    # Load model
    model = ConvTasNet.load_model(args.model_path)
 #   print(model)
    model.eval()
    if args.use_cuda:
        model.cuda(0)

    # Load data
    dataset = AudioDataset(args.data_dir, args.batch_size,
                           sample_rate=args.sample_rate, segment=2)
    data_loader = AudioDataLoader(dataset, batch_size=1, num_workers=2)

    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            print(i)
            # Get batch data
            padded_mixture, mixture_lengths, padded_source = data
            if args.use_cuda:
                padded_mixture = padded_mixture.cuda(0)
                mixture_lengths = mixture_lengths.cuda(0)
            # Forward
            estimate_source ,s_embed  = model(padded_mixture)  # [B, C, T],#[B,N,K,E] 
          #  print(estimate_source.shape)
           # embid = (model.separator.network[2][7])(padded_mixture)
          #  print(embid)
            '''
            embeddings = s_embed[0].data.cpu().numpy()
            embedding = (embeddings.reshape((1,-1,20)))[0]
            number = sourceNumEsti2(embedding)
            numberEsti.append(number)
            '''
           # print(estimate_source)
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            # Remove padding and flat
            mixture = remove_pad(padded_mixture, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)
           # print(max_snr.item())
            # NOTE: use reorder estimate source
            estimate_source = remove_pad(reorder_estimate_source,
                                         mixture_lengths)
           # print((estimate_source[0].shape))
            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                print("Utt", total_cnt + 1)
                # Compute SDRi
                if args.cal_sdr:
                    avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                    total_SDRi += avg_SDRi
                    print("\tSDRi={0:.2f}".format(avg_SDRi))
                # Compute SI-SNRi
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                #avg_SISNRiPit,a,b = cal_si_snr_with_pit(torch.from_numpy(src_ref), torch.from_numpy(src_est),length)
                print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                total_SISNRi += (avg_SISNRi)
                #total_SNRiPitNum += avg_SISNRiPit.numpy()
                total_cnt += 1
            
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))
    print("speaker:2 ./ClustertrainTFSE1New/final_paper_2_3_2chobatch6.pth.tar")
   
    return numberEsti


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:python
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2 #
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = (sisnr1-sisnr1b + sisnr2-sisnr2b) / 2  #1
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr

def sourceNumEsti(v):
    sortNum = []
    cov = np.cov(v.T)
    covNew = cov[0:cov.shape[0]-1,0:cov.shape[0]-1]
    eigenValue,featureVector = np.linalg.eig(covNew)
   #print(eigenValue)
    eigenValue = np.array(eigenValue)
    featureVector = np.array(featureVector)
    normEig = np.linalg.norm(eigenValue)
  # print(normEig)
    eigNew = np.append(eigenValue, normEig)
    diag = np.diag(eigNew)
    #print(eigNew)
    transV = np.zeros(cov.shape)
    transV[0:cov.shape[0]-1,0:cov.shape[0]-1] = featureVector
    transV[cov.shape[0]-1,cov.shape[0]-1] = 1
    covTrans =np.dot(np.dot(transV.T,cov),transV)
    covD = np.dot(np.dot(diag,covTrans),np.linalg.inv(diag))
    covD = np.dot(np.dot(diag,covD),np.linalg.inv(diag))
    covD = np.dot(np.dot(diag,covD),np.linalg.inv(diag))
    Ger = covD[0:cov.shape[0]-1, cov.shape[0]-1]
    listGer = (np.abs(Ger)).tolist()
    listGer.sort(reverse = True)
    '''
    for i in range(0,len(listGer)-1):
        sortNum.append( listGer[i]/listGer[i+1])  
    '''     
 #  averageListGer = sum(i for i in listGer[0:int(len(listGer))])/(len(listGer))

    #numberSource = sortNum.index(max(sortNum))+1
   # numberSource = sum(i>=averageListGer*0.0001 for i in listGer)
    numberSource = sum(i>=400 for i in listGer)
    if(numberSource !=2):
        print(listGer)
        #print(averageListGer)
        print(numberSource)

    return numberSource   #sortNum.index(max(sortNum))+1
def sourceNumEsti2(v):
    cov = np.cov(v.T)
    m,n = np.linalg.eig(cov)
    listGer = (np.abs(m)).tolist()
    listGer.sort(reverse = True)
    averageListGer = sum(i for i in listGer[0:int(len(listGer))])/(len(listGer))
    numberSource = sum(i>=averageListGer*1.55 for i in m)
    if(numberSource !=2):
        print(listGer)
        print(averageListGer)
        print(numberSource)
    return numberSource   #s

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    numberEsti = evaluate(args)
    print(sum(j==2 for j in numberEsti))
