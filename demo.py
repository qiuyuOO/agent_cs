import polars as pl
from awpy import Demo

dem = Demo(r"D:\5E_cs2_demo\g161-20260827151717641063023_de_dust2\g161-20260827151717641063023_de_dust2.dem")
dem.parse()


def build_round_summaries(dem):
    round_list = []

    for row in dem.rounds.iter_rows(named=True):
        round_list.append(row)

    return round_list

round_summaries = build_round_summaries(dem)

print(f"总回合数: {len(round_summaries)}")
print("\n前3个回合数据:")
for i in range(min(3, len(round_summaries))):
    print(round_summaries[i])