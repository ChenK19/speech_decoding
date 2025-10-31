'''
Author: WSF
email: 18875355021@163.com

date:2025-10
updte:

Copyright (c) 2025 WSF. All Rights Reserved.

'''


from neuracle_lib.readbdfdata import readbdfdata
from tkinter import filedialog
from tkinter import *
import numpy as np
import os
import mne
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt



def check_files_format(path):
     filename = []
     pathname = []
     if len(path) == 0:
          raise TypeError('please select valid file')

     elif len(path) == 1:
          (temppathname, tempfilename) = os.path.split(path[0])
          if 'edf' in tempfilename:
               filename.append(tempfilename)
               pathname.append(temppathname)
               return filename, pathname
          elif 'bdf' in tempfilename:
               raise TypeError('unsupport only one neuracle-bdf file')
          else:
               raise TypeError('not support such file format')

     else:
          temp = []
          temppathname = r''
          evtfile = []
          idx = np.zeros((len(path) - 1,))
          for i, ele in enumerate(path):
               (temppathname, tempfilename) = os.path.split(ele)
               if 'data' in tempfilename:
                    temp.append(tempfilename)
                    if len(tempfilename.split('.')) > 2:
                         try:
                              idx[i] = (int(tempfilename.split('.')[1]))
                         except:
                              raise TypeError('no such kind file')
                    else:
                         idx[i] = 0
               elif 'evt' in tempfilename:
                    evtfile.append(tempfilename)

          pathname.append(temppathname)
          datafile = [temp[i] for i in np.argsort(idx)]

          if len(evtfile) == 0:
               raise TypeError('not found evt.bdf file')

          if len(datafile) == 0:
               raise TypeError('not found data.bdf file')
          elif len(datafile) > 1:
               print('current readbdfdata() only support continue one data.bdf ')
               return filename, pathname
          else:
               filename.append(datafile[0])
               filename.append(evtfile[0])
               return filename, pathname

def chans_pick(rawdata, chans=None):
    """
    Parameters
    ----------
    rawdata : array, [chans_num, class_num, trial_num, sample_num]
    chans : list, channels name

    Returns
    -------
    data : array, [chans_num, class_num, trial_num, sample_num]

    10-10 system
    CHANNELS = [
        'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3',
        'F1','FZ','F2','F4','F6','F8','FT7','FC5',
        'FC3','FC1','FCZ','FC2','FC4','FC6','FC8','T7',
        'C5','C3','C1','CZ','C2','C4','C6','T8',
        'M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4',
        'CP6','TP8','M2','P7','P5','P3','P1','PZ',
        'P2','P4','P6','P8','PO7','PO5','PO3','POZ',
        'PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'
    ]
    """
    CHANNELS = ['FPZ', 'FP1', 'FP2', 'AF3', 'AF4', 'AF7', 'AF8', 'FZ', 
                 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
                 'FCZ', 'FC1', 'FC2', 'FC3','FC4', 'FC5', 'FC6', 'FT7', 
                 'FT8', 'CZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
                'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 
                'TP7', 'TP8', 'PZ', 'P3', 'P4', 'P5', 'P6', 'P7', 
                'P8', 'POZ', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
                 'OZ', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']

    idx_loc = []
    if isinstance(chans, list):
        for chans_value in chans:
            idx_loc.append(CHANNELS.index(chans_value.upper()))

    data = rawdata[idx_loc, ...] if idx_loc else rawdata
    return data



if __name__ == '__main__':

    '''
    mVEP-P300特征离线提取脚本
    首先将数据降采样至200Hz，选取16个导联（F3/4、Fz、C3/4、Cz、T7/8、P3/4、Pz、P7/8、PO7/8和Oz），
    然后通过巴特沃斯滤波器进行[1Hz, 10Hz]的带通滤波，再降采样至20Hz，最后提取刺激开始后[50ms, 800ms]时间窗内的特征。
    超声脑机接口 ： 比较新的脑机领域 ， 人机融合
    
    '''

    ###Extract data
    marker_id = [1, 2, 3, 4, 5, 14, 15, 16]  # event markers
    target_id = [6, 7, 8, 9, 10, 11, 12, 13]  # target markers
    stim_random_sequence = np.array([2, 7, 4, 1, 6, 3, 0, 5])
    chans = ['F3', 'F4', 'FZ', 'C3', 'C4', 'CZ', 'T7', 'T8', 'P3', 'P4', 'P7', 'P8',
              'PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    Left_chans = [6, 8, 10, 13, 14] 
    right_chans = [7, 9, 11, 16, 17]
    choose_Pz_channel = [12]
    choose_Cz_channel = [5]


    chans = [  'C3', 'C4', 'CZ',  'P3', 'P4', 'P7', 'P8', 
              'PZ','PO5', 'PO3', 'POZ', 'PO4', 'PO6','O1', 'OZ', 'O2']
    Left_chans = [3, 5, 8, 9] 
    right_chans = [4, 6, 11, 12]
    choose_Pz_channel = [7]
    choose_Cz_channel = [2]


    channel_num= 64
    direction_angle = ['Right', 'Right-Top', 'Top', 'Left-Top', 'Left', 'Left-Bottom', 'Bottom', 'Right-Bottom']
    block_num = 4  
    singleblock_trial_num = 5
    trial_markers = 5*8+1
    tmin, tmax = -0.2, 0.8  # epoch time range
    down_Fs_rate = 5  #
    down_Fs = int(1000/down_Fs_rate)
    filter_num = 4
    n_samples = int((tmax - tmin) * down_Fs)  # number of samples per epoch
    file_num = 3
    # Filter bank setting
    w_pass_2d = np.array([[ 1, 1, 1, 1], [ 10, 14, 20, 30]])
    w_stop_2d = np.array([[ 0.1, 0.1, 0.1, 0.1], [ 12, 16, 22, 32]])

    pathname = ['C:/Users/mac/Desktop/screen/mvep_p300/wsf-data/wzz/mvep/1', 
                'C:/Users/mac/Desktop/screen/mvep_p300/wsf-data/wzz/mvep/2',
                'C:/Users/mac/Desktop/screen/mvep_p300/wsf-data/wzz/mvep/3']
    filename = ['data.bdf', 'evt.bdf']
    target_Alltrial = np.zeros( len(marker_id) * block_num * file_num )
    label_TargetOrNontarget_Alltrial = np.zeros( [len(target_Alltrial), int(trial_markers-1)] )
    All_trial_raw = np.zeros([len(target_Alltrial), int(trial_markers-1), channel_num, n_samples])

     ## select file
    # root = Tk()
    # root.withdraw()
    # ## select bdf or edf file
    # path = filedialog.askopenfilenames(initialdir='/', title='Select two bdf files',
    #                                     filetypes=(("two bdf files", "*.bdf"), ("one edf files", "*.edf")))
    # ## check files format
    # filename, pathname = check_files_format(path)
    # ## parse data

    for file_i in range(file_num):
        print('Processing file:', pathname[file_i] + '/' + filename[0])
        # read data
        raw = readbdfdata(filename, pathname[file_i:file_i+1])
        raw.load_data()
        # filter
        raw= raw.filter(l_freq=1, h_freq=10, method='fir')
        if not raw:
            print('raw is empty')
        else:
            print( raw.info['ch_names'])

        Fs = int(raw.info['sfreq'])
        print('Original Sampling Rate:', Fs)
        # extract epochs
        events, events_id = mne.events_from_annotations(raw)
        picks = mne.pick_types(raw.info, emg=False, eeg=True, stim=False, eog=False)
        picks_ch_names = [raw.ch_names[i] for i in picks]

        rawData = {}
        dataDic = {}
        event_label = np.array(events_id.keys())
        # print(event_label)
        for marker_i in range(1,len(marker_id)+1):
            # print(marker_i)
            rawData[marker_i] = mne.Epochs(raw, events=events, event_id = marker_id[marker_i-1], tmin=tmin, picks=picks,tmax=tmax,
                                baseline=None, preload=True).get_data() 
            # downsampling / by 'down'
            dataDic[marker_i] = mne.filter.resample(rawData[marker_i], down = down_Fs_rate, n_jobs = 1)
            sfreq = int(Fs / 5)

        data = list(dataDic.values())
        down_dataArray = np.array(data[0:len(marker_id)])    # [class_num, trial_num, chans_num, sample_num]
        print('down_dataArray shape:', down_dataArray.shape)

        # extract traget
        print('Event length:', len(events))
        target_trial = np.zeros( len(marker_id) * block_num ) 
        for i_target in range(len(marker_id) * block_num):
            target_trial[i_target] = int(events[i_target * trial_markers, 2] - 5) 
        print('target_trial:', target_trial)
        label_TargetOrNontarget_trial = np.zeros( [len(target_trial), int(trial_markers-1)] )
        for i_trial in range(len(target_trial)):
            for i_single_trial in range(singleblock_trial_num):
                nontarget_target_label = np.ones( len(marker_id) ) * 0  # non-target
                index_target = np.where( stim_random_sequence == int(target_trial[i_trial]-1) )[0]  # stim_random_sequence = np.array([2, 7, 4, 1, 6, 3, 0, 5])
                nontarget_target_label[index_target] = 1  # target
                # nontarget_target_label[ int(target_trial[i_trial]-1) ] = 1  # target
                label_TargetOrNontarget_trial[i_trial, i_single_trial*len(marker_id):(i_single_trial+1)*len(marker_id)] = nontarget_target_label
        print('label_TargetOrNontarget_trial shape:', label_TargetOrNontarget_trial.shape)

        # a = np.array([1,2,3,4,5,6,7,8])
        # b = np.array([7,6,5,4,3,2,1,0])
        # print("Output:",a[b]) 

        # reshape down_dataArray
        trial_raw = np.zeros([len(target_trial), int(trial_markers-1), down_dataArray.shape[2], down_dataArray.shape[3]])
        for i_trial in range(len(target_trial)):
            for i_single_trial in range(singleblock_trial_num):
                trial_raw[i_trial, 
                        ( i_single_trial *len(marker_id) ) :  ( (i_single_trial+1) *len(marker_id) )] = down_dataArray[stim_random_sequence ,
                                                                                    i_trial*singleblock_trial_num + i_single_trial]
        print('trial_raw shape:', trial_raw.shape)  # [total_trial_num, singleblock_trial_num*class_num, chans_num, sample_num]
        
        target_Alltrial[ file_i * len(target_trial) : (file_i+1) * len(target_trial) ] = target_trial
        label_TargetOrNontarget_Alltrial[ file_i * len(target_trial) : (file_i+1) * len(target_trial), : ] = label_TargetOrNontarget_trial
        All_trial_raw[ file_i * len(target_trial) : (file_i+1) * len(target_trial), :, :, : ] = trial_raw

    print('target_Alltrial shape:', target_Alltrial.shape)  # [total_trial_num]
    print('label_TargetOrNontarget_Alltrial shape:', label_TargetOrNontarget_Alltrial.shape)  # [total_trial_num, singleblock_trial_num*class_num]
    print('All_trial_raw shape:', All_trial_raw.shape)  # [total_trial_num, singleblock_trial_num*class_num, chans_num, sample_num]

    # n_class  target vs non-target
    Nclass_target_trial = np.zeros([len(marker_id) , block_num*file_num])
    Nclass_label_TargetOrNontarget_trial = np.zeros([len(marker_id) , block_num*file_num, int(trial_markers-1)])
    Nclass_trial_raw = np.zeros([len(marker_id) , block_num*file_num, int(trial_markers-1), down_dataArray.shape[2], down_dataArray.shape[3]])
    for i_class in range(len(marker_id)):
        index_class = np.where( target_Alltrial == (i_class+1) )[0]
        Nclass_target_trial[i_class, :] = target_Alltrial[index_class]
        Nclass_label_TargetOrNontarget_trial[i_class, :, :] = label_TargetOrNontarget_Alltrial[index_class, :]
        Nclass_trial_raw[i_class, :, :, :, :] = All_trial_raw[index_class, :, :, :]
    print('Nclass_trial_raw shape:', Nclass_trial_raw.shape)  # [class_num, block_num, singleblock_trial_num*class_num, chans_num, sample_num]

    # pick chans
    Nclass_trial_raw = np.transpose(Nclass_trial_raw, [3, 0, 1, 2, 4])    #  channel_num, class_num, block_num, singleblock_trial_num*class_num, sample_num
    Nclass_trial_raw_Chans = chans_pick(Nclass_trial_raw, chans)
    Nclass_trial_raw_Chans = np.transpose(Nclass_trial_raw_Chans, [1, 2, 3, 0, 4])  # class_num, block_num, singleblock_trial_num*class_num, channel_num , sample_num
    print('Nclass_trial_raw_Chans shape:', Nclass_trial_raw_Chans.shape)  # [class_num, block_num, singleblock_trial_num*class_num, channel_num , sample_num]

    Nclass_target_chans =  np.zeros( [len(marker_id) , block_num*file_num, singleblock_trial_num,  Nclass_trial_raw_Chans.shape[3], Nclass_trial_raw_Chans.shape[4]] )
    Nclass_nontarget_chans =  np.zeros( [len(marker_id) , block_num*file_num, singleblock_trial_num*(len(marker_id)-1),  Nclass_trial_raw_Chans.shape[3], Nclass_trial_raw_Chans.shape[4]] )
    for i_class in range(len(marker_id)):
        target_index = np.where( Nclass_label_TargetOrNontarget_trial[i_class, 0] == 1 )[0]
        nontarget_index = np.where( Nclass_label_TargetOrNontarget_trial[i_class, 0] == 0 )[0]
        Nclass_target_chans[i_class] = np.transpose( Nclass_trial_raw_Chans[i_class, :,  target_index ], (1,0,2,3) )
        Nclass_nontarget_chans[i_class] = np.transpose( Nclass_trial_raw_Chans[i_class, :,  nontarget_index ], (1,0,2,3) )
    

    
    mean_Nclass_target_chans = np.mean(np.mean(Nclass_target_chans, axis=2, keepdims=False) , axis=1, keepdims=False)
    mean_Nclass_nontarget_chans = np.mean(np.mean(Nclass_nontarget_chans, axis=2, keepdims=False) , axis=1, keepdims=False)
    print('mean_Nclass_target_chans shape:', Nclass_target_chans.shape, mean_Nclass_target_chans.shape)  # [class_num, channel_num , sample_num]
    print('mean_Nclass_nontarget_chans shape:', Nclass_nontarget_chans.shape, mean_Nclass_nontarget_chans.shape)  # [class_num, channel_num , sample_num]

    # plot average evoked potential
    plt.rcParams.update({
    "font.size": 12,          # 全局基础字体大小
    "axes.titlesize": 14,     # 标题字体大小
    "axes.labelsize": 14,     # 坐标轴标签字体大小
    "xtick.labelsize": 12,    # X轴刻度标签字体大小
    "ytick.labelsize": 12     # Y轴刻度标签字体大小
    })

    plt.subplots(4,2,figsize=(20, 11), constrained_layout=True)  
    time_axis = np.arange(tmin*1000, tmax*1000, 1000/sfreq)  # ms

    for i_class in range(len(marker_id)):
        plt.subplot(4,2,i_class+1)
        
        target_data = mean_Nclass_target_chans[i_class, :, :] 
        nontarget_data = mean_Nclass_nontarget_chans[i_class, :, :]  

        left_target_data = target_data[Left_chans, :]
        right_target_data = target_data[right_chans, :]
        
        target_mean = np.mean(target_data, axis=0)  
        nontarget_mean = np.mean(nontarget_data, axis=0)  
        
        for i_channel in range(target_data.shape[0]):
            plt.plot(time_axis, target_data[i_channel, :], 
                    color='lightcoral', linewidth=0.5, alpha=0.3)
        
        for i_channel in range(nontarget_data.shape[0]):
            plt.plot(time_axis, nontarget_data[i_channel, :], 
                    color='lightblue', linewidth=0.5, alpha=0.3)

        
        plt.plot(time_axis, target_mean, 
                color='red', linewidth=1.5, alpha=1.0, label='Target Average')
        
        plt.plot(time_axis, nontarget_mean, 
                color='blue', linewidth=1.5, alpha=1.0, label='Non-Target Average')
        
        plt.title(direction_angle[i_class])
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.grid(alpha=0.3)  
        plt.xlim([-200, 800])
        if i_class in [6]:
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude (uV)')
            plt.legend()


    plt.suptitle('Average Evoked Potentials for Target and Non-Target Stimuli\n(Thin lines: Individual channels, Thick lines: Average)', 
                fontsize=14)  
    # plt.show()

    plt.subplots(4,2,figsize=(20, 11), constrained_layout=True)  
    time_axis = np.arange(tmin*1000, tmax*1000, 1000/sfreq)  # ms

    for i_class in range(len(marker_id)):
        plt.subplot(4,2,i_class+1)
        
        target_data = mean_Nclass_target_chans[i_class, :, :] 
        nontarget_data = mean_Nclass_nontarget_chans[i_class, :, :]  

        left_target_data = target_data[Left_chans, :]
        right_target_data = target_data[right_chans, :]
        
        target_mean = np.mean(target_data, axis=0)  
        nontarget_mean = np.mean(nontarget_data, axis=0)  
        
        for i_channel in range(left_target_data.shape[0]):
            if i_channel ==0:
                plt.plot(time_axis, left_target_data[i_channel, :], 
                    color='orange', linewidth=0.5, alpha=0.5, label='Left Channels' )
            else:
                plt.plot(time_axis, left_target_data[i_channel, :], 
                    color='orange', linewidth=0.5, alpha=0.5 )
            
        for i_channel in range(right_target_data.shape[0]):
            if i_channel ==0:
                plt.plot(time_axis, right_target_data[i_channel, :], 
                    color='deepskyblue', linewidth=0.5, alpha=0.5, label='Right Channels' )
            else:   
                plt.plot(time_axis, right_target_data[i_channel, :], 
                    color='deepskyblue', linewidth=0.5, alpha=0.5 )
        
        plt.plot(time_axis, target_mean, 
                color='red', linewidth=1.5, alpha=1.0, label='Target Average')
        
        plt.plot(time_axis, nontarget_mean, 
                color='blue', linewidth=1.5, alpha=1.0, label='Non-Target Average')
        
        plt.title(direction_angle[i_class])
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.grid(alpha=0.3)  
        plt.xlim([-200, 800])
        if i_class in [6]:
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude (uV)')
            plt.legend()


    plt.suptitle('Average Evoked Potentials for Target and Non-Target Stimuli\n(Thin lines: Individual channels, Thick lines: Average)', 
                fontsize=14)  

    # single channel——trial Pz Cz target vs non-target
    Nclass_target_Pz = Nclass_target_chans[:, :, :, choose_Pz_channel, : ].squeeze()
    Nclass_nontarget_Pz = Nclass_nontarget_chans[:, :, :, choose_Pz_channel, : ].squeeze()
    Nclass_target_Cz = Nclass_target_chans[:, :, :, choose_Cz_channel, : ].squeeze()
    Nclass_nontarget_Cz = Nclass_nontarget_chans[:, :, :, choose_Cz_channel, : ].squeeze()
    print('Nclass_target_Pz shape:', Nclass_target_Pz.shape)  # [class_num, block_num, singleblock_trial_num, sample_num]
    print('Nclass_nontarget_Pz shape:', Nclass_nontarget_Pz.shape)  # [class_num, block_num, singleblock_trial_num, sample_num]

    # reshape Nclass_target_Pz
    Nclass_target_Pz_reshaped = np.zeros( [len(marker_id), Nclass_target_Pz.shape[1]*Nclass_target_Pz.shape[2], Nclass_target_Pz.shape[3]] )
    Nclass_nontarget_Pz_reshaped = np.zeros( [len(marker_id), Nclass_nontarget_Pz.shape[1]*Nclass_nontarget_Pz.shape[2], Nclass_nontarget_Pz.shape[3]] )
    Nclass_target_Cz_reshaped = np.zeros( [len(marker_id), Nclass_target_Cz.shape[1]*Nclass_target_Cz.shape[2], Nclass_target_Cz.shape[3]] )
    Nclass_nontarget_Cz_reshaped = np.zeros( [len(marker_id), Nclass_nontarget_Cz.shape[1]*Nclass_nontarget_Cz.shape[2], Nclass_nontarget_Cz.shape[3]] )
    for i_class in range(len(marker_id)):
        Nclass_target_Pz_reshaped[i_class, :, :] = np.reshape( Nclass_target_Pz[i_class, :, :, :], ( Nclass_target_Pz.shape[1]*Nclass_target_Pz.shape[2], Nclass_target_Pz.shape[3] ) )
        Nclass_nontarget_Pz_reshaped[i_class, :, :] = np.reshape( Nclass_nontarget_Pz[i_class, :, :, :], ( Nclass_nontarget_Pz.shape[1]*Nclass_nontarget_Pz.shape[2], Nclass_nontarget_Pz.shape[3] ) )
        Nclass_target_Cz_reshaped[i_class, :, :] = np.reshape( Nclass_target_Cz[i_class, :, :, :], ( Nclass_target_Cz.shape[1]*Nclass_target_Cz.shape[2], Nclass_target_Cz.shape[3] ) )
        Nclass_nontarget_Cz_reshaped[i_class, :, :] = np.reshape( Nclass_nontarget_Cz[i_class, :, :, :], ( Nclass_nontarget_Cz.shape[1]*Nclass_nontarget_Cz.shape[2], Nclass_nontarget_Cz.shape[3] ) )
    print('Nclass_target_Pz_reshaped shape:', Nclass_target_Pz_reshaped.shape)  # [class_num, total_target_trial_num, sample_num]
    print('Nclass_nontarget_Pz_reshaped shape:', Nclass_nontarget_Pz_reshaped.shape)  # [class_num, total_nontarget_trial_num, sample_num]

    # # plot Pz target vs non-target
    # plt.subplots(4,2,figsize=(20, 11), constrained_layout=True)  
    # time_axis = np.arange(tmin*1000, tmax*1000, 1000/sfreq)  # ms   
    # for i_class in range(len(marker_id)):
    #     plt.subplot(4,2,i_class+1)
        
    #     target_data = Nclass_target_Pz_reshaped[i_class, :, :] 
    #     nontarget_data = Nclass_nontarget_Pz_reshaped[i_class, :, :]    
    #     target_mean = np.mean(target_data, axis=0)  
    #     nontarget_mean = np.mean(nontarget_data, axis=0)
    #     # for i_trial in range(target_data.shape[0]):
    #     #     plt.plot(time_axis, target_data[i_trial, :], 
    #     #             color='lightcoral', linewidth=0.5, alpha=0.3)
    #     # for i_trial in range(nontarget_data.shape[0]):
    #     #     plt.plot(time_axis, nontarget_data[i_trial, :], 
    #     #             color='lightblue', linewidth=0.5, alpha=0.3)
    #     plt.plot(time_axis, target_mean, 
    #             color='red', linewidth=1.5, alpha=1.0, label='Target Average')
    #     plt.plot(time_axis, nontarget_mean, 
    #             color='blue', linewidth=1.5, alpha=1.0, label='Non-Target Average')
    #     plt.title(direction_angle[i_class] + ' - Pz Channel')
    #     plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
    #     plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    #     plt.grid(alpha=0.3)
    #     plt.xlim([-200, 800])
    #     if i_class in [6]:
    #         plt.xlabel('Time (ms)')
    #         plt.ylabel('Amplitude (uV)')
    #         plt.legend()    
    # plt.suptitle('Pz Channel: Evoked Potentials for Target and Non-Target Stimuli\n(Thin lines: Individual trials, Thick lines: Average)', 
    #         fontsize=14)
    
    # # plot Cz target vs non-target
    # plt.subplots(4,2,figsize=(20, 11), constrained_layout=True)  
    # time_axis = np.arange(tmin*1000, tmax*1000, 1000/sfreq)  # ms   
    # for i_class in range(len(marker_id)):
    #     plt.subplot(4,2,i_class+1)
        
    #     target_data = Nclass_target_Cz_reshaped[i_class, :, :] 
    #     nontarget_data = Nclass_nontarget_Cz_reshaped[i_class, :, :] 
    #     target_mean = np.mean(target_data, axis=0)  
    #     nontarget_mean = np.mean(nontarget_data, axis=0)
    #     # for i_trial in range(target_data.shape[0]):
    #     #     plt.plot(time_axis, target_data[i_trial, :], 
    #     #             color='lightcoral', linewidth=0.5, alpha=0.3)
    #     # for i_trial in range(nontarget_data.shape[0]):
    #     #     plt.plot(time_axis, nontarget_data[i_trial, :], 
    #     #             color='lightblue', linewidth=0.5, alpha=0.3)
    #     plt.plot(time_axis, target_mean, 
    #             color='red', linewidth=1.5, alpha=1.0, label='Target Average')
    #     plt.plot(time_axis, nontarget_mean, 
    #             color='blue', linewidth=1.5, alpha=1.0, label='Non-Target Average')
    #     plt.title(direction_angle[i_class] + ' - Cz Channel')
    #     plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
    #     plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    #     plt.grid(alpha=0.3) 
    #     plt.xlim([-200, 800])
    #     if i_class in [6]:
    #         plt.xlabel('Time (ms)')
    #         plt.ylabel('Amplitude (uV)')
    #         plt.legend()
    # plt.suptitle('Cz Channel: Evoked Potentials for Target and Non-Target Stimuli\n(Thin lines: Individual trials, Thick lines: Average)', 
    #         fontsize=14)
    
    
    plt.show()

