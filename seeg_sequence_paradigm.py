#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计声母 韵母 与 音调（数字声调）
依赖：pypinyin
pip install pypinyin
"""

import csv
import re
import sys
from collections import Counter
try:
    from pypinyin import pinyin, Style
except ImportError:
    print("请先安装依赖：pip install pypinyin", file=sys.stderr)
    sys.exit(1)

# ---------- 句子列表（你之前提供的 100 条） ----------
sentences = [
"周末去公园散步",
"记得给植物浇水",
"提前准备演讲稿",
"今天天气真好",
"把钥匙放好吧",
"我们一起吃饭",
"朋友来我家玩",
"他每天跑步忙",
"你可以坐这儿",
"妈妈做好晚饭",
"请把窗户打开",
"别忘了带伞",
"孩子在院里玩",
"我去超市买菜",
"老师布置作业",
"把杯子放回桌",
"她喜欢喝咖啡",
"周末去看电影",
"他会弹吉他了",
"请把灯关掉",
"我把票订好了",
"门要记得上锁",
"把照片打印好",
"我们去图书馆",
"爸妈要来探望",
"请把垃圾扔好",
"她每天练字帖",
"车票已经退了",
"把窗帘拉上吧",
"邻居送来水果",
"明天早起锻炼",
"我想学做蛋糕",
"请把文件备份",
"周末一起野餐",
"小狗在追球跑",
"别忘记吃药哦",
"她在准备考试",
"我们去拍照片",
"把钥匙收好点",
"今天晚饭真香",
"他喜欢种花草",
"把衣服拿去洗",
"我已发邮件给你",
"请把日程确认",
"他们在讨论计划",
"我们约好见面",
"把报表发给我",
"她学会做蛋糕",
"请把椅子摆好",
"小明在操场跑步",
"我们一起看书",
"把门铃修好啦",
"他为你做晚饭",
"请把手机调静音",
"妈妈在阳台浇花",
"我准备去体检",
"把鞋子摆整齐",
"大家一起打扫",
"她写了一封信",
"请把窗台擦干净",
"我们明天出发",
"把行李收拾好",
"他每天练瑜伽",
"请把灯光调暗",
"孩子的作业检查",
"我们去拜访老人",
"把文件分类整理",
"她喜欢拍风景",
"请把水壶放回去",
"我们约在车站见",
"把菜切好放冰箱",
"他修好了自行车",
"请把会议记录好",
"我把钥匙借给你",
"她在练习唱歌",
"把门窗都锁好",
"我们一起做晚餐",
"请在群里提醒大家",
"他每天写日记本",
"把报表核对清楚",
"我们计划志愿活动",
"请把冰箱清理掉",
"她为朋友买礼物",
"把孩子衣服熨好",
"我们在路口等你",
"请把仪器擦干净",
"他每天学编程",
"把照片做成相册",
"我们去海边散步",
"请把密码保存好",
"她学会骑自行车",
"把旧衣捐给需要人",
"我们周五吃火锅",
"请把投影打开好",
"他准备了小礼物",
"把花浇水别忘记",
"我们去看展览吧",
"请把会议时间发群",
"她每天练习书法",
"把窗户纱网修好",
"我们周末去爬山",
"周末去郊外踏青",
"今天天气真好",
"明早别忘带伞",
"请把门锁好",
"我们一起吃饭",
"妈妈做好晚饭",
"把钥匙放好处",
"孩子在院子玩",
"朋友来家里坐",
"他每天跑三圈",
"你可以坐这里",
"请把窗户打开",
"别忘了带水杯",
"我去超市买菜",
"老师布置作业了",
"把衣服收拾好",
"她喜欢喝咖啡",
"周末去看电影",
"他会弹吉他啦",
"请把灯关掉吧",
"票已经订好了",
"门口有人来访",
"把照片打印出",
"我们去图书馆读",
"爸妈周末来访",
"请把垃圾分类",
"她每天练字帖",
"车票退好了没",
"把窗帘拉上去",
"邻居送水果来",
"明早要早起跑",
"我想学做面包",
"请把文件备份",
"周末去野外游",
"小狗在追球跑",
"别忘记吃药哦",
"她在准备考试",
"我们去拍照片吧",
"把钥匙收起来",
"今晚晚饭真香",
"他喜欢种花草",
"把衣服拿去洗",
"我已发邮件给你",
"请把日程确认",
"他们在商量计划",
"我们约好见面",
"把报表发给我",
"她学会做蛋糕",
"请把椅子摆好",
"小明在操场跑步",
"我们一起读书吧",
"把门铃修一下",
"他为你做晚饭",
"请把手机调静音",
"妈妈在阳台浇花",
"我准备去体检了",
"把鞋子摆整齐",
"大家一同打扫屋",
"她写了一封信",
"请把窗台擦干净",
"我们明天就出发",
"把行李收拾好吧",
"他每天练瑜伽哦",
"请把灯光调暗些",
"孩子作业要检查",
"我们去拜访老人",
"把文件分类整理",
"她喜欢拍风景照",
"请把水壶放回去",
"我们车站见面吧",
"把菜切好放冰箱",
"他修好了自行车",
"请把会议记录好",
"我把钥匙借给你",
"她在练习唱歌呢",
"把门窗都锁好了",
"我们一起做晚餐",
"请在群里提醒下",
"他每天写读书记",
"把报表核对清楚",
"我们计划做公益",
"请把冰箱清理好",
"她为朋友买礼物",
"把孩子衣服熨好",
"我们在路口等你",
"请把仪器擦干净",
"他每天学编程呢",
"把照片做成相册",
"我们去海边散步",
"请把密码保存好",
"她学会骑自行车",
"把旧衣捐给需人",
"我们周五吃火锅",
"请把投影打开好",
"他准备了小礼物",
"把花浇水别忘记",
"我们去看展览吧",
"请把会议时间发群",
"她每天练习书法",
"把窗户纱网修好",
"我们周末去爬山",
"早上喝杯暖牛奶"
]

# ---------- 工具与计数器 ----------
def is_cjk(ch):
    return '\u4e00' <= ch <= '\u9fff'

initials_counter = Counter()
finals_counter = Counter()
tones_counter = Counter()

per_sentence = []  # 可选，存每句的细节

# Regex 用于从 TONE3 拼音（如 zhong1）提取尾部数字声调
tone_re = re.compile(r'([1-5])$')

# ---------- 逐句逐字统计 ----------
for sent in sentences:
    chars = [c for c in sent if is_cjk(c)]
    initials_seq = []
    finals_seq = []
    tones_seq = []
    for ch in chars:
        # 使用 TONE3 得到带数字的拼音 比如 zhong1
        pinyin_t3 = pinyin(ch, style=Style.TONE3, strict=False, heteronym=False)
        pinyin_init = pinyin(ch, style=Style.INITIALS, strict=False, heteronym=False)
        pinyin_fin = pinyin(ch, style=Style.FINALS, strict=False, heteronym=False)

        # pinyin 返回列表结构 [[str]]，我们取第一个元素
        p_t3 = pinyin_t3[0][0] if pinyin_t3 and pinyin_t3[0] else ""
        p_init = pinyin_init[0][0] if pinyin_init and pinyin_init[0] else ""
        p_fin = pinyin_fin[0][0] if pinyin_fin and pinyin_fin[0] else ""

        # 规范化：空声母标为 (none)
        init_norm = p_init if p_init else "(none)"
        fin_norm = p_fin if p_fin else "(none)"

        # 提取声调数字（1-5），若无匹配则标为 0（表示未知/未标注）
        m = tone_re.search(p_t3)
        tone = m.group(1) if m else "0"
        if not m:
            print(1)

        initials_seq.append(init_norm)
        finals_seq.append(fin_norm)
        tones_seq.append(tone)

        initials_counter[init_norm] += 1
        finals_counter[fin_norm] += 1
        tones_counter[tone] += 1

    per_sentence.append({
        "sentence": sent,
        "chars": "".join(chars),
        "initials": initials_seq,
        "finals": finals_seq,
        "tones": tones_seq
    })

# ---------- 输出与导出 ----------
total = sum(initials_counter.values())  # 总音节数

def print_counter(title, counter):
    print(f"=== {title} ===")
    for k, v in counter.most_common():
        pct = v / total * 100 if total else 0
        print(f"{k:8s}  {v:4d}  ({pct:5.2f}%)")
    print()

print(f"总音节数 {total}\n")
print_counter("声母统计 (initials)", initials_counter)
print_counter("韵母统计 (finals)", finals_counter)

# 音调统计（按数字 1 2 3 4 5 或 0 未识别）
print("=== 音调统计 (tones 数字表示) ===")
for tone, cnt in tones_counter.most_common():
    pct = cnt / total * 100 if total else 0
    label = tone
    if tone == "0":
        label = "0(unknown)"
    print(f"{label:8s}  {cnt:4d}  ({pct:5.2f}%)")
print()

# 导出 CSV 文件
with open("initials_counts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["initial", "count", "percentage"])
    for k, v in initials_counter.most_common():
        w.writerow([k, v, f"{v/total:.6f}" if total else "0"])

with open("finals_counts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["final", "count", "percentage"])
    for k, v in finals_counter.most_common():
        w.writerow([k, v, f"{v/total:.6f}" if total else "0"])

with open("tones_counts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["tone", "count", "percentage"])
    for k, v in tones_counter.most_common():
        w.writerow([k, v, f"{v/total:.6f}" if total else "0"])

print("已导出 initials_counts.csv finals_counts.csv tones_counts.csv")

# 可选：把每句的逐字拼音信息也写出以便逐句核对（如需可以打开）
# with open("per_sentence_details.csv", "w", newline="", encoding="utf-8") as f:
#     w = csv.writer(f)
#     w.writerow(["sentence", "chars", "initials", "finals", "tones"])
#     for item in per_sentence:
#         w.writerow([
#             item["sentence"],
#             item["chars"],
#             " ".join(item["initials"]),
#             " ".join(item["finals"]),
#             " ".join(item["tones"])
#         ])
