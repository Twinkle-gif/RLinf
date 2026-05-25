"""Robust scalar dump from a TFRecord events file: skip bad records, no full read."""
import sys, struct, zlib
from collections import defaultdict
from tensorboard.compat.proto import event_pb2

path = sys.argv[1]
series = defaultdict(list)

with open(path, 'rb') as f:
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        length = struct.unpack('<Q', hdr)[0]
        f.read(4)  # crc_len, skip
        buf = f.read(length)
        if len(buf) < length:
            break
        f.read(4)  # crc_data, skip
        try:
            ev = event_pb2.Event()
            ev.ParseFromString(buf)
        except Exception:
            continue
        if not ev.summary or not ev.summary.value:
            continue
        for v in ev.summary.value:
            if v.HasField('simple_value'):
                series[v.tag].append((ev.step, v.simple_value))

print('NUM_TAGS', len(series))

def stat(name):
    arr = sorted(series.get(name, []))
    v = [a[1] for a in arr]
    n = len(v)
    if n == 0: return
    Q = max(1, n // 5)
    parts = [sum(v[i*Q:(i+1)*Q])/max(1, len(v[i*Q:(i+1)*Q])) for i in range(5)]
    print(f"{name:42s} N={n:5d} LAST={v[-1]:9.4f} parts={[f'{x:7.3f}' for x in parts]} MIN={min(v):8.3f} MAX={max(v):8.3f}")

focus = [
    'env/success_once','env/reward','env/episode_len',
    'env/mean_reward_approach','env/mean_reward_placement','env/mean_reward_total',
    'env/mean_reward_disturbance','env/mean_reward_success_bonus',
    'rollout/returns_mean','rollout/rewards',
    'rollout/advantages_max','rollout/advantages_min',
    'train/critic/value_loss','train/critic/explained_variance',
    'train/actor/policy_loss','train/actor/policy_loss_abs','train/actor/ratio_abs',
    'train/actor/approx_kl','train/actor/clip_fraction','train/actor/grad_norm',
    'train/actor/entropy_loss','train/actor/lr',
]
for t in focus:
    if t in series: stat(t)

# Also print first 20 steps of key signals to detect odd-even pattern
for k in ('env/reward','env/mean_reward_approach','env/mean_reward_placement','rollout/returns_mean'):
    arr = sorted(series.get(k, []))
    if not arr: continue
    print(f'--- {k} first 20 ---')
    for x in arr[:20]:
        print(f'  step={x[0]:4d} v={x[1]:.5f}')