import mne

from neuracle_lib.readbdfdata import readbdfdata
from tkinter import filedialog
from tkinter import *
import numpy as np
import os

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

if __name__ == '__main__':

    pathname = ['Data/20251030eeg-language/ck3',
    'Data/20251030eeg-language/ck4',
    'Data/20251030eeg-language/ck5',
    'Data/20251030eeg-language/ck6',
    'Data/20251030eeg-language/ck7',
    'Data/20251030eeg-language/ck8',
    'Data/20251030eeg-language/ck9',
    'Data/20251030eeg-language/ck10',
    'Data/20251030eeg-language/ck11',
    'Data/20251030eeg-language/ck12']

    filename = ['data.bdf', 'evt.bdf']

    # marker_id = [1, 2, 3, 4] 
    # M1 - fixation - M2 - sti(reading) - fixation - M3 - sti - fix -M3 - sti - M4
    # M1 - 1S       - M2 - 0.5S         - 0.5S     - M3 - 1S  - 0.5S-M3 - 1S  - M4

    marker_id = [3]
    tmin, tmax = 0, 0.8
    raw = readbdfdata(filename, pathname)
    raw.load_data()
    events, events_id = mne.events_from_annotations(raw)
    picks = mne.pick_types(raw.info, emg=False, eeg=True, stim=False, eog=False)

    ch_names = raw.info['ch_names']
    print(ch_names)

    rawData = {}
    for marker_i in range(1,len(marker_id)+1):
        # print(marker_i)
        rawData[marker_i] = mne.Epochs(raw, events=events, event_id = marker_id[marker_i-1], tmin=tmin, picks=picks,tmax=tmax,
                            baseline=None, preload=True).get_data() 

    # sf_pathname = ['Data/20251027_mandarin_test/胜福_test']
    # sf_raw = readbdfdata(filename, sf_pathname)
    # sf_raw.load_data()
    # sf_events, sf_events_id = mne.events_from_annotations(sf_raw)
    
    print('debug')