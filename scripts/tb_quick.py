import sys
from collections import defaultdict
from tensorboard.compat.proto import event_pb2
import struct

path = sys.argv[1]
with open(path, 'rb') as f:
    data = f.read()

i = 0
N = len(data)
series = defaultdict(list)
while i + 12 <= N:
    length = struct.unpack('<Q', data[i:i+8])[0]
    i += 12
    if i + length + 4 > N:
        break
    buf = data[i:i+length]
    i += length + 4
    ev = event_pb2.Event()
    try:
        ev.ParseFromString(buf)
    except Exception:
        continue
    if not ev.summary or not ev.summary.value:
        continue
    for v in ev.summary.value:
        if v.HasField('simple_value'):
            series[v.tag].append((ev.step, v.simple_value))

print('NUM_TAGS', len(series))

focus = [
    'env/success_once','env/reward','env/return','env/episode_len',
    'env/mean_reward_approach','env/mean_reward_placement','env/mean_reward_total',
    'env/mean_reward_disturbance','env/mean_reward_success_bonus',
    'env/return_reward_approach','env/return_reward_placement','env/return_reward_total',
    'rollout/returns_mean','rollout/rewards','rollout/advantages_mean',
    'rollout/advantages_max','rollout/advantages_min',
    'train/critic/value_loss','train/critic/explained_variance','train/critic/value_clip_ratio',
    'train/actor/policy_loss','train/actor/policy_loss_abs','train/actor/ratio_abs',
    'train/actor/approx_kl','train/actor/clip_fraction','train/actor/grad_norm',
    'train/actor/entropy_loss','train/actor/total_loss','train/actor/lr',
]

def stat(name, arr):
    arr = sorted(arr)
    v = [a[1] for a in arr]
    st = [a[0] for a in arr]
    n = len(v)
    if n == 0: return
    Q = max(1, n // 4)
    e = sum(v[:Q])/Q
    m = sum(v[Q:2*Q])/Q
    l = sum(v[2*Q:3*Q])/Q
    end = sum(v[3*Q:])/max(1, n-3*Q)
    print(f"{name:45s} N={n:5d} STEP_LAST={st[-1]:5d} LAST={v[-1]:.5f} Q1={e:.5f} Q2={m:.5f} Q3={l:.5f} Q4={end:.5f} MIN={min(v):.5f} MAX={max(v):.5f}")

for t in focus:
    if t in series:
        stat(t, series[t])

def dump(name, head=30, tail=10):
    s = sorted(series.get(name, []))
    if not s: return
    print(f'--- {name} first {head} + last {tail} ---')
    for x in s[:head]: print(' s', x[0], 'v', round(x[1], 5))
    print('  ...')
    for x in s[-tail:]: print(' s', x[0], 'v', round(x[1], 5))

dump('env/mean_reward_placement')
dump('env/mean_reward_approach')
dump('env/reward', head=30, tail=5)
dump('rollout/returns_mean', head=30, tail=5)
dump('train/actor/ratio_abs')
dump('train/actor/approx_kl')
dump('env/success_once')
dump('train/critic/explained_variance', head=20, tail=5)