from mcp.server.mcpserver import MCPServer
from awpy import Demo
import polars as pl


mcp = MCPServer(name='demo MCP Server')

@mcp.tool(
    name='get_kill',
    description=(
        "获取 CS2 demo 的所有击杀记录。"
        "返回 list[dict], 每条击杀 dict 含字段: "
        "attacker(击杀者名), victim(被击杀者名), assister(助攻者名,可为 null), "
        "weapon(武器名如 ak47/awp), "
        "headshot(是否爆头 bool), "
        "attacker_blind(攻击者是否被闪 bool), "
        "through_smoke(是否穿烟击杀 bool), penetrated(是否穿透击杀 bool), "
        "noscope(是否盲狙 bool), attacker_in_air(是否空中击杀 bool), "
        "distance(击杀距离,单位 CS2 距离), "
        "attacker_health(攻击者击杀时血量), victim_health(受害者血量,通常 0), "
        "attacker_place(攻击者位置名如 'LongA'), victim_place(受害者位置名), "
        "round(回合数 1-N), tick(事件 tick 编号, tick/64=距 demo 开头秒数). "
        "注意: awpy 1.x 无 'victim_blind' 字段(原代码误用,已移除); 若需判断受害者是否被闪, "
        "应结合 get_prop 的 flash_groups 数据按 tick 邻近度配对 player_blind 事件。"
        "无击杀返回空 list."
    )
)
def get_kill(demo_path:str)->list:
    """
    获取 CS2 demo 的所有击杀记录 (含助攻/爆头/穿烟等标志位)。

    Args:
        demo_path: demo 文件绝对路径 传入的参数的路径要有转义字符

    Returns:
        list[dict]: 每条 dict 结构见 @mcp.tool description;
        字段名对应 awpy 1.x 的 dem.kills DataFrame, 已重命名为下划线风格。
    """
    demo = Demo(demo_path)
    demo.parse()

    data = []



    for kill in demo.kills.iter_rows(named=True):
        kill_info = {
            "attacker": kill["attacker_name"],
            "victim": kill["victim_name"],
            "assister": kill["assister_name"],

            "weapon": kill["weapon"],

            "headshot": kill["headshot"],
            "attacker_blind": kill["attackerblind"],
            "through_smoke": kill["thrusmoke"],
            "penetrated": kill["penetrated"],
            "noscope": kill["noscope"],
            "attacker_in_air": kill["attackerinair"],

            "distance": kill["distance"],

            "attacker_health": kill["attacker_health"],
            "victim_health": kill["victim_health"],

            "attacker_place": kill["attacker_place"],
            "victim_place": kill["victim_place"],

            "round": kill["round_num"],
            "tick": kill["tick"]
        }
        data.append(kill_info)

    return data


@mcp.tool(
    name='get_prop',
    description=(
        "获取 CS2 demo 中每回合的道具使用情况(烟雾/闪光/HE/燃烧弹/诱饵)。"
        "返回 dict[round_num -> dict], 每回合 dict 含三个 list: 'throws'/'damages'/'flash_groups'. "
        "throws 元素字段: type(smoke/flash/he/molotov/incendiary/decoy), thrower(投掷者), "
        "side(t/ct), tick(投掷 tick), x/y/z(爆开位置), round_num; "
        "smoke 额外有 end_tick 和 duration_sec(秒, 烟雾持续时长, 最后一回合可能为 null). "
        "damages 元素字段: type(he/molotov/incendiary), weapon(原始武器名 hegrenade/inferno), "
        "attacker, attacker_side, victim, victim_side, dmg_health(单次伤害血量), tick, round_num. "
        "flash_groups 元素字段(每颗闪光一组): thrower, thrower_side, tick(爆开 tick), "
        "x/y/z(爆开位置), round_num, victims(list[dict: name/side/blind_duration/tick]), "
        "victim_count(这颗闪闪到几个人), total_blind_time(总闪眼秒数). "
        "回合无道具数据则对应 list 为空. 24 回合比赛通常 30 个 round_num 键."
    )
)
def get_prop(demo_path: str)->dict:
    """
    获取 CS2 demo 中每回合的道具使用情况(含投掷/伤害/闪光受害者分组)。

    记录:
        1. 谁投掷了什么道具 (5 种: smoke/flash/he/molotov/incendiary/decoy)
        2. HE / Molotov / Incendiary 对谁造成多少伤害
        3. Flash 闪到了谁, 以及闪光持续时间 (按 entityid 分组)
        4. Smoke 投掷信息 (含 end_tick + duration_sec), Decoy 数量统计

    Args:
        demo_path: demo 文件绝对路径 传入的参数的路径要有转义字符

    Returns:
        dict[round_num -> dict]: 结构见 @mcp.tool description.
        每回合 dict 形如 {"throws": [...], "damages": [...], "flash_groups": [...]}.
    """

    demo = Demo(demo_path)

    extended_events = demo.default_events + [
        "player_blind",
        "decoy_detonate",
    ]
    demo.parse(events=extended_events)

    rounds_df = demo.rounds.select(["round_num", "start", "end"]).sort("round_num")
    rounds = rounds_df.to_dicts()

    def tick_to_round(tick):
        for r in rounds:
            if r["start"] <= tick <= r["end"]:
                return r["round_num"]
        return None

    throws = []

    for s in demo.smokes.select([
        "start_tick", "end_tick", "thrower_name", "thrower_side", "X", "Y", "Z", "round_num"
    ]).to_dicts():
        start = s["start_tick"]
        end = s["end_tick"]
        duration = round((end - start) / 64, 2) if (start is not None and end is not None) else None
        throws.append({
            "type": "smoke",
            "thrower": s["thrower_name"],
            "side": s["thrower_side"],
            "tick": s["start_tick"],
            "end_tick": s["end_tick"],
            "duration_sec": duration,
            "x": s["X"], "y": s["Y"], "z": s["Z"],
            "round_num": s["round_num"],
        })

    for m in demo.infernos.select([
        "start_tick", "thrower_name", "thrower_side", "X", "Y", "Z", "round_num"
    ]).to_dicts():
        throws.append({
            "type": "molotov" if m["thrower_side"] == "t" else "incendiary",
            "thrower": m["thrower_name"],
            "side": m["thrower_side"],
            "tick": m["start_tick"],
            "x": m["X"], "y": m["Y"], "z": m["Z"],
            "round_num": m["round_num"],
        })

    for f in demo.events["flashbang_detonate"].iter_rows(named=True):
        throws.append({
            "type": "flash",
            "thrower": f["user_name"],
            "side": f["user_side"],
            "tick": f["tick"],
            "x": f["x"], "y": f["y"], "z": f["z"],
            "round_num": tick_to_round(f["tick"]),
        })

    for h in demo.events["hegrenade_detonate"].iter_rows(named=True):
        throws.append({
            "type": "hegrenade",
            "thrower": h["user_name"],
            "side": h["user_side"],
            "tick": h["tick"],
            "x": h["x"], "y": h["y"], "z": h["z"],
            "round_num": tick_to_round(h["tick"]),
        })

    for d in demo.events["decoy_detonate"].iter_rows(named=True):
        throws.append({
            "type": "decoy",
            "thrower": d["user_name"],
            "side": d["user_side"],
            "tick": d["tick"],
            "x": d["x"], "y": d["y"], "z": d["z"],
            "round_num": tick_to_round(d["tick"]),
        })


    util_damages = []
    util_dmg_df = demo.damages.filter(
        pl.col("weapon").is_in(["hegrenade", "inferno"])
    ).select([
        "tick", "round_num", "weapon",
        "attacker_name", "attacker_side",
        "victim_name", "victim_side",
        "dmg_health",
    ])

    for d in util_dmg_df.to_dicts():
        weapon = d["weapon"]
        if weapon == "hegrenade":
            dmg_type = "hegrenade"
        else:
            dmg_type = "molotov" if d["attacker_side"] == "t" else "incendiary"
        util_damages.append({
            "weapon": weapon,
            "type": dmg_type,
            "attacker": d["attacker_name"],
            "attacker_side": d["attacker_side"],
            "victim": d["victim_name"],
            "victim_side": d["victim_side"],
            "dmg_health": d["dmg_health"],
            "tick": d["tick"],
            "round_num": d["round_num"],
        })
        #print(f"{d["attacker_name"]}扔出{weapon}对{d["victim_name"]}造成{d["dmg_health"]}点伤害")

    blinds_by_entity = {}
    for b in demo.events["player_blind"].iter_rows(named=True):
        eid = b["entityid"]
        blinds_by_entity.setdefault(eid, []).append({
            "name": b["user_name"],
            "side": b["user_side"],
            "blind_duration": round(b["blind_duration"], 2),
            "tick": b["tick"],
        })

    flash_groups = []
    for f in demo.events["flashbang_detonate"].iter_rows(named=True):
        eid = f["entityid"]
        victims = blinds_by_entity.get(eid, [])
        flash_groups.append({
            "entityid": eid,
            "thrower": f["user_name"],
            "thrower_side": f["user_side"],
            "tick": f["tick"],              # 爆开 tick
            "x": f["x"], "y": f["y"], "z": f["z"],   # 爆开位置
            "round_num": tick_to_round(f["tick"]),
            "victims": victims,
            "victim_count": len(victims),
            "total_blind_time": round(sum(v["blind_duration"] for v in victims), 2),
        })

    result = {}
    for r in rounds:
        rn = r["round_num"]
        result[rn] = {
            "throws": [t for t in throws if t["round_num"] == rn],
            "damages": [d for d in util_damages if d["round_num"] == rn],
            "flash_groups": [g for g in flash_groups if g["round_num"] == rn],
        }

    return result

if __name__ == "__main__":
    mcp.run()