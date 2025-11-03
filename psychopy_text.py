# -*- coding: utf-8 -*-
from psychopy import visual, core, event, clock, parallel
from screeninfo import get_monitors
import random
import numpy as np
from neuracle_lib.triggerBox import TriggerBox,PackageSensorPara,TriggerIn
import time

def get_refresh_rate():
    # Windows
    for monitor in get_monitors():
        if monitor.is_primary and hasattr(monitor, 'refresh_rate'):
            return monitor.refresh_rate
    return 120  # Default refresh rate if not found

# ========= 可调参数 =========
FULLSCREEN    = True
WIN_SIZE      = (1280*2, 800*2)
BG_GRAY       = -0.2
FIX_SECONDS   = 1.5
STIM_SECONDS  = 3.0
HANZI_TEXT    = ["妈", "嘛", "马", "骂", "咪", "迷", "米", "蜜"]
PINYIN_TEXT   = ["mā", "má", "mǎ", "mà", "mī", "mí", "mǐ", "mì"]
HANZI_FONT    = "Songti SC"
PINYIN_FONT   = "Helvetica"
HANZI_HEIGHT  = 100
PINYIN_HEIGHT = 80
LINE_GAP_PX   = 120
# 进度条（白线）参数
BAR_FULL_W    = 100    # 初始总宽
BAR_H         = 10      # 厚度
BAR_Y         = -200   # 位置
BAR_COLOR     = [1, 1, 1]
# ===========================

win = visual.Window(
    size=WIN_SIZE,
    units="pix",
    color=[BG_GRAY, BG_GRAY, BG_GRAY],
    fullscr=FULLSCREEN,
    waitBlanking=True,
    useFBO=False,
    allowGUI=False,
)

logfile = f"psychopy_text_2025_10_31_ck1.log"
logfile_handle = open(logfile, "a")
print(f"[INFO] Log file: {logfile}")

logfile_handle.write(f"\n\n=== New Session: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

# 预热并测 FPS
_pre = core.Clock()
while _pre.getTime() < 1.0:
    win.flip()
# FPS = win.getActualFrameRate(nIdentical=60, nMaxFrames=240, nWarmUpFrames=60, threshold=1) or 60.0
FPS = get_refresh_rate()
print(f"[INFO] Refresh ≈ {FPS:.2f} Hz")

# 固定点与刺激
fix = visual.TextStim(win, text="+", color="white", height=100)
hanzi_cue  = [visual.TextStim(win, text=x,  color="red", height=HANZI_HEIGHT,
                         font=HANZI_FONT,  pos=(0,  LINE_GAP_PX/2)) for x in HANZI_TEXT]
pinyin_cue = [visual.TextStim(win, text=x, color="red", height=PINYIN_HEIGHT,
                         font=PINYIN_FONT, pos=(0, -LINE_GAP_PX/2)) for x in PINYIN_TEXT]
hanzi  = [visual.TextStim(win, text=x,  color="white", height=HANZI_HEIGHT,
                         font=HANZI_FONT,  pos=(0,  LINE_GAP_PX/2)) for x in HANZI_TEXT]
pinyin = [visual.TextStim(win, text=x, color="white", height=PINYIN_HEIGHT,
                         font=PINYIN_FONT, pos=(0, -LINE_GAP_PX/2)) for x in PINYIN_TEXT]

# # === 用 ShapeStim 实现白色进度线 ===
# def rect_vertices_centered(w, h):
#     hw, hh = w/2.0, h/2.0
#     return [(-hw, -hh), ( hw, -hh), ( hw,  hh), (-hw,  hh)]

def rect_vertices_centered(w, h):
    hw, hh = w/2.0, h/2.0
    return [(-hw, -hh), ( hw, -hh), ( hw,  hh), (-hw,  hh)]

def rect_vertices_left_anchored(w, h):
    # 左侧为锚点（左边不动，向右收缩/扩展）
    return [(0, -h/2.0), (w, -h/2.0), (w, h/2.0), (0, h/2.0)]

BAR_BG_COLOR  = [-0.6, -0.6, -0.6]
BAR_FG_COLOR  = [1.0, 1.0, 1.0]

bar_bg = visual.ShapeStim(
    win, vertices=rect_vertices_centered(BAR_FULL_W, BAR_H),
    pos=(0, BAR_Y), fillColor=BAR_BG_COLOR, lineColor=None,
    closeShape=True, interpolate=False, opacity=1.0, units="pix",
)
bar_fg = visual.ShapeStim(
    win, vertices=rect_vertices_left_anchored(BAR_FULL_W, BAR_H),
    pos=(-BAR_FULL_W/2.0, BAR_Y),   # 左边对齐到背景条左侧
    fillColor=BAR_FG_COLOR, lineColor=None,
    closeShape=True, interpolate=False, opacity=1.0, units="pix",
)

def show_for(stims, seconds, with_progress=False):
    frames = int(round(seconds * FPS))
    clk = core.Clock()

    # 每段时窗单独创建一条“白线”
    if with_progress:
        # bar_line = visual.ShapeStim(
        #     win,
        #     vertices=rect_vertices_centered(BAR_FULL_W, BAR_H),
        #     pos=(0, BAR_Y),
        #     fillColor=BAR_COLOR, lineColor=None,
        #     closeShape=True, interpolate=False, units="pix"
        # )
        bar_bg = visual.ShapeStim(
            win, vertices=rect_vertices_centered(BAR_FULL_W, BAR_H),
            pos=(0, BAR_Y), fillColor=BAR_BG_COLOR, lineColor=None,
            closeShape=True, interpolate=False, opacity=1.0, units="pix",
        )
        bar_fg = visual.ShapeStim(
            win, vertices=rect_vertices_left_anchored(BAR_FULL_W, BAR_H),
            pos=(-BAR_FULL_W/2.0, BAR_Y),   # 左边对齐到背景条左侧
            fillColor=BAR_FG_COLOR, lineColor=None,
            closeShape=True, interpolate=False, opacity=1.0, units="pix",
        )

    for i in range(frames):
        if 'escape' in event.getKeys():
            win.close(); core.quit()

        for s in stims:
            s.draw()

        if with_progress:
            # frac = max(0.0, 1.0 - (i / max(1, frames - 1)))
            # cur_w = max(1.0, BAR_FULL_W * frac)  # 避免严格到 0 宽
            # # 用 setVertices 显式更新，触发重建
            # bar_line.setVertices(rect_vertices_centered(cur_w, BAR_H), log=False)
            # bar_line.draw()
            bar_bg.draw()
            frac = max(0.0, (i / max(1, frames - 1)))  # 剩余比例
            cur_w = BAR_FULL_W * frac
            if cur_w > 0:
                bar_fg.vertices = rect_vertices_left_anchored(cur_w, BAR_H)
                bar_fg.draw()

        win.flip()
        # if clk.getTime() >= seconds:
        #     break

N_repeats_per_block = 3
N_trails_per_block = len(hanzi) * N_repeats_per_block

random_idx_list = [_ for _ in range(len(hanzi))] * N_repeats_per_block
np.random.shuffle(random_idx_list)

triggerin = TriggerIn("COM3")
flag = triggerin.validate_device()
if flag:
    print("TriggerIn device is valid")
else:
    print("device not found")

stim_clock = clock.Clock()

for i in range(N_trails_per_block):

    random_idx = random_idx_list[i]
    print(f"Trial {i+1}/{N_trails_per_block}, Stimulus: {HANZI_TEXT[random_idx]} ({PINYIN_TEXT[random_idx]})")
    logfile_handle.write(f"Trial {i+1}/{N_trails_per_block}, Stimulus: {HANZI_TEXT[random_idx]} ({PINYIN_TEXT[random_idx]})\n")

    stim_clock.reset()
    triggerin.output_event_data(1)
    show_for([fix], 1.4, with_progress=False)
    triggerin.output_event_data(2)
    show_for([hanzi_cue[random_idx], pinyin_cue[random_idx]], 1.0, with_progress=False)
    triggerin.output_event_data(5)
    show_for([fix], 0.8, with_progress=False)
    triggerin.output_event_data(3)
    show_for([hanzi[random_idx], pinyin[random_idx]], 1.2, with_progress=True)
    triggerin.output_event_data(5)
    show_for([fix], 0.8, with_progress=False)
    triggerin.output_event_data(3)
    show_for([hanzi[random_idx], pinyin[random_idx]], 1.2, with_progress=True)
    triggerin.output_event_data(4)
    print("Stimulus time: ", stim_clock.getTime())  # 打印该目标刺激段耗时
    logfile_handle.write(f"Stimulus time: {stim_clock.getTime()}\n")
    # show_for([fix], FIX_SECONDS, with_progress=False)
    # show_for([hanzi[random_idx], pinyin[random_idx]], STIM_SECONDS, with_progress=True)

# # === 1 s 固定点 + 2 s 刺激（带进度线） ===
# show_for([fix], FIX_SECONDS, with_progress=True)
# show_for([hanzi, pinyin], STIM_SECONDS, with_progress=True)
# # === 1 s 固定点 + 2 s 刺激（带进度线） ===
# show_for([fix], FIX_SECONDS, with_progress=True)
# show_for([hanzi, pinyin], STIM_SECONDS, with_progress=True)
# # === 1 s 固定点 + 2 s 刺激（带进度线） ===
# show_for([fix], FIX_SECONDS, with_progress=True)
# show_for([hanzi, pinyin], STIM_SECONDS, with_progress=True)
# # === 1 s 固定点 + 2 s 刺激（带进度线） ===
# show_for([fix], FIX_SECONDS, with_progress=True)
# show_for([hanzi, pinyin], STIM_SECONDS, with_progress=True)
# # === 1 s 固定点 + 2 s 刺激（带进度线） ===
# show_for([fix], FIX_SECONDS, with_progress=True)
# show_for([hanzi, pinyin], STIM_SECONDS, with_progress=True)
# # === 1 s 固定点 + 2 s 刺激（带进度线） ===
# show_for([fix], FIX_SECONDS, with_progress=True)
# show_for([hanzi, pinyin], STIM_SECONDS, with_progress=True)

win.close()
core.quit()
triggerin.closeSerial()