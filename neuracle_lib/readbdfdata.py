

# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: FANG Junying, fangjunying@neuracle.cn
#
# Versions:
# 	v0.1: 2018-08-14, orignal
#   v0.2: 2019-11-04, update read evt.bdf annotation method
#   v1.0: 2020-12-12, update event, available mne
#   v1.1: 2024-06-19, create mne RAW object

# Copyright (c) 2016 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/

import mne,os,re
import numpy as np

def read_annotations_bdf(annotations):
    pat = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
    if isinstance(annotations, str):
        with open(annotations, encoding='latin-1') as annot_file:
            triggers = re.findall(pat, annot_file.read())
    else:
        tals = bytearray()
        for chan in annotations:
            this_chan = chan.ravel()
            if this_chan.dtype == np.int32:  # BDF
                this_chan.dtype = np.uint8
                this_chan = this_chan.reshape(-1, 4)
                # Why only keep the first 3 bytes as BDF values
                # are stored with 24 bits (not 32)
                this_chan = this_chan[:, :3].ravel()
                for s in this_chan:
                    tals.extend(s)
            else:
                for s in this_chan:
                    i = int(s)
                    tals.extend(np.uint8([i % 256, i // 256]))

        # use of latin-1 because characters are only encoded for the first 256
        # code points and utf-8 can triggers an "invalid continuation byte"
        # error
        triggers = re.findall(pat, tals.decode('latin-1'))

    events = []
    for ev in triggers:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0
        for description in ev[3].split('\x14')[1:]:
            if description:
                events.append([onset, duration, description])
    return zip(*events) if events else (list(), list(), list())


def readbdfdata(filename, pathname):
    """
    Parameters
    ----------
    filename: list of str
    pathname: list of str

    Return:
    ----------
    raw, mne Raw object (with correct units in Volt)
    """
    import os
    import mne
    import numpy as np

    if 'edf' in filename[0]:  # DSI
        raw = mne.io.read_raw_edf(os.path.join(pathname[0], filename[0]), preload=True)
    else:  # Neuracle
        # ✅ 1. 正确加载数据，强制 preload=True
        raw = mne.io.read_raw_bdf(os.path.join(pathname[0], 'data.bdf'), preload=True)
        fs = raw.info['sfreq']

        # ✅ 2. 检查是否未标定
        data = raw.get_data()
        std = np.std(data)
        if std > 1e-3:  # 振幅太大说明未缩放
            print("⚠️ 检测到未标定的Neuracle数据，执行缩放 (ADC→Volt)")
            scale = 3.125e-8  # 0.03125 µV per bit
            data = data * scale
            info = raw.info.copy()
            raw = mne.io.RawArray(data, info)
        else:
            print("✅ 已检测到正确单位 (Volt)")

        # ✅ 3. 尝试加载事件
        try:
            annotationData = mne.io.read_raw_bdf(os.path.join(pathname[0], 'evt.bdf'), preload=True)
            try:
                tal_data = annotationData._read_segment_file([], [], 0, 0, int(annotationData.n_times), None, None)
                print('mne version <= 0.20')
            except Exception:
                idx = np.empty(0, int)
                tal_data = annotationData._read_segment_file(
                    np.empty((0, annotationData.n_times)),
                    idx,
                    0,
                    0,
                    int(annotationData.n_times),
                    np.ones((len(idx), 1)),
                    None
                )
                print('mne version > 0.20')
            onset, duration, description = read_annotations_bdf(tal_data[0])
            evt_annotations = mne.Annotations(onset=onset, duration=duration, description=description)
            raw.set_annotations(evt_annotations)
        except FileNotFoundError:
            print('⚠️ 未找到事件文件 (evt.bdf)')
    return raw