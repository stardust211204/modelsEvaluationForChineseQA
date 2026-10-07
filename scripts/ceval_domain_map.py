"""C-Eval 52 学科 → MMLU-ProX 14 领域 的映射（终版）。

设计原则（2026-10-05 定）
    1. **目标领域只用 MMLU-ProX 原有的 14 个** —— 不新增，否则两个题库无法合并对比。
    2. **分类的作用对所有模型一致** —— 所以分类不必很细；真正要避免的是两件事：
         (a) **种类偏离太大** —— 不能把数学题归到生物里
         (b) **不要把难度搞混** —— C-Eval 的 `other` 一律视作难题（职业层）
    3. 两者口径本来就不一致（MMLU 是国外题集、C-Eval 是国内题集）：
       例如「毛泽东思想」实质是马克思主义的一种，归入 philosophy 是合理的。
    4. **资格考试类逐题判断** —— 公务员/教师资格/导游的题目内容杂，
       按学科名整体归类会让化学题落进「其他」，所以按 **instance_id** 逐题指定。

几个刻意的决定
    * `geography` → `other`。原以为该排除（题集没有地理领域），但**查了 MMLU-ProX 的
      `other` 之后改了主意**：它的 46 题里是会计税务(11)、美国政治法律(7)、社会科学(6)、
      流行文化杂项(6)……**本质是原 MMLU 的兜底类，不是某个学科**。
      既然如此，地理放进 `other` 比丢掉更合适 —— 保留数据，且逻辑一致。
    * `art_studies` / `education_science` → `psychology`（教育科学属教育心理范畴）
    * `mao_zedong_thought` / `ideological_and_moral_cultivation` / 政治 → `philosophy`
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from meval import build_set, paths

EXCLUDED = "__EXCLUDED__"       # 不列入任何领域统计

# --------------------------------------------------------------------------
# 一、整学科映射
# --------------------------------------------------------------------------
MAP: dict[str, str] = {
    # ---- 初中 ----
    "middle_school_mathematics": "math",
    "middle_school_physics": "physics",
    "middle_school_chemistry": "chemistry",
    "middle_school_biology": "biology",
    "middle_school_history": "history",
    "middle_school_geography": "other",           # 地理 -> other（见下方说明）
    "middle_school_politics": "philosophy",       # 政治 -> 哲学（按口径一致原则）
    # ---- 高中 ----
    "high_school_mathematics": "math",
    "high_school_physics": "physics",
    "high_school_chemistry": "chemistry",
    "high_school_biology": "biology",
    "high_school_history": "history",
    "high_school_geography": "other",             # 地理 -> other
    "high_school_politics": "philosophy",
    "high_school_chinese": "other",
    # ---- 大学 ----
    "college_economics": "economics",
    "college_programming": "computer science",
    "college_chemistry": "chemistry",
    "college_physics": "physics",
    # ---- 职业：财务会计 ----
    "tax_accountant": "business",
    "accountant": "business",
    "business_administration": "business",
    # ---- 职业：医学健康 ----
    "physician": "health",
    "veterinary_medicine": "health",
    "clinical_medicine": "health",
    "basic_medicine": "health",
    "sports_science": "health",
    # ---- 职业：工程 ----
    "urban_and_rural_planner": "engineering",
    "electrical_engineer": "engineering",
    "fire_engineer": "engineering",
    "metrology_engineer": "engineering",
    "environmental_impact_assessment_engineer": "engineering",
    # ---- 职业：法律 ----
    "law": "law",
    "legal_professional": "law",
    # ---- 职业：哲学 ----
    "logic": "philosophy",
    "marxism": "philosophy",
    "mao_zedong_thought": "philosophy",
    "ideological_and_moral_cultivation": "philosophy",
    # ---- 职业：历史 ----
    "modern_chinese_history": "history",
    # ---- 职业：计算机 ----
    "operating_system": "computer science",
    "computer_architecture": "computer science",
    "computer_network": "computer science",
    # ---- 职业：生物 ----
    "plant_protection": "biology",
    # ---- 职业：数学 ----
    "advanced_mathematics": "math",
    "probability_and_statistics": "math",
    "discrete_mathematics": "math",
    # ---- 职业：教育/艺术/文学 -> 按决定归 psychology 或 other ----
    "art_studies": "psychology",
    "education_science": "psychology",
    "chinese_language_and_literature": "other",
    # 以下三个**逐题判断**，见 BY_ID
    "civil_servant": "other",
    "teacher_qualification": "other",
    "professional_tour_guide": "other",
}

# --------------------------------------------------------------------------
# 二、资格考试类：逐题映射（覆盖上面的整学科映射）
# --------------------------------------------------------------------------
BY_ID: dict[str, str] = {
    # ---- civil_servant（20 题）：混着化学/数学/法律/经济/天文 ----
    "ceval-civil_servant-46": "law",            # 执法效果与法律文本
    "ceval-civil_servant-252": "chemistry",     # 化学与日常生活
    "ceval-civil_servant-15": "law",            # 「平反」定义判断
    "ceval-civil_servant-412": "math",          # 数列 2,1,4,3,8,5
    "ceval-civil_servant-382": "economics",     # 橄榄型收入分配
    "ceval-civil_servant-115": "physics",       # 宇宙中最基本的天体
    "ceval-civil_servant-388": "chemistry",     # 环境化学效应
    "ceval-civil_servant-281": "math",          # 蔬菜消耗应用题
    # 其余 12 题是言语理解/类比推理/文学常识 -> other（保持默认）

    # ---- teacher_qualification（19 题）：绝大部分是教育学/心理学 ----
    "ceval-teacher_qualification-253": "law",              # 课堂伤害赔偿责任
    "ceval-teacher_qualification-178": "law",              # 教育法规（体育课替代自习）
    "ceval-teacher_qualification-165": "history",          # 凿壁偷光
    "ceval-teacher_qualification-75": "history",           # 春秋五霸
    "ceval-teacher_qualification-115": "computer science", # Excel 图表
    "ceval-teacher_qualification-321": "health",           # 维生素 D 缺乏
    # 其余 13 题是教育学/教育心理学 -> psychology
    **{f"ceval-teacher_qualification-{i}": "psychology" for i in
       (200, 286, 307, 339, 100, 243, 223, 216, 142, 261, 296, 124, 274)},

    # ---- professional_tour_guide（13 题）：历史/宗教/艺术/常识 ----
    "ceval-professional_tour_guide-14": "history",   # 上海世博会
    "ceval-professional_tour_guide-95": "history",   # 古代昆明族
    "ceval-professional_tour_guide-74": "history",   # 抗战局面
    "ceval-professional_tour_guide-128": "history",  # 古代国号由来
    "ceval-professional_tour_guide-146": "history",  # 白马寺
    # 其余 8 题是茶文化/宗教/绘画/文学 -> other
}

PROX14 = ["math", "physics", "chemistry", "law", "engineering", "other", "economics",
          "business", "psychology", "biology", "health", "philosophy",
          "computer science", "history"]

# C-Eval 的 other 一律视作**难题**（职业层）
HARD_LEVELS = {"职业"}


def lvl(subject: str) -> str:
    if subject.startswith("middle_school"):
        return "初中"
    if subject.startswith("high_school"):
        return "高中"
    if subject.startswith("college"):
        return "大学"
    return "职业"


def domain_of(inst) -> str:
    return BY_ID.get(inst.id) or MAP[inst.stratum["subject"]]


def main() -> int:
    ev = build_set.load_frozen(paths.data() / "eval_set.jsonl")
    ce = build_set.load_frozen(paths.data() / "ceval_set.jsonl")

    prox = Counter(i.stratum["subject"] for i in ev)
    unmapped = {i.stratum["subject"] for i in ce} - set(MAP)
    if unmapped:
        print(f"  ⚠️ 未映射: {sorted(unmapped)}")
        return 1

    cell: dict[str, Counter] = defaultdict(Counter)
    n_excl = 0
    for i in ce:
        d = domain_of(i)
        if d == EXCLUDED:
            n_excl += 1
            continue
        cell[d][lvl(i.stratum["subject"])] += 1
    # other 组内实际难度分布（按决定：other 一律视作难题）
    if "other" in cell:
        cell["other"]["职业"] = sum(v for k, v in cell["other"].items()) - cell["other"]["职业"]
        for k in list(cell["other"]):
            if k != "职业":
                del cell["other"][k]

    print("=" * 100)
    print("  C-Eval 按 MMLU-ProX 14 领域聚合（终版）")
    print("=" * 100)
    print(f"  {'领域':<18}{'C-Eval':>7}{'ProX':>7}{'合计':>7}   难度构成")
    tc = tp = 0
    for d in PROX14:
        c = cell.get(d, Counter())
        n = sum(c.values())
        p = prox.get(d, 0)
        tc += n
        tp += p
        comp = " ".join(f"{k}{v}" for k, v in c.items() if v) or "—"
        flag = "  ⚠️少" if 0 < n < 20 else ("  ❌无" if n == 0 else "")
        print(f"  {d:<18}{n:>7}{p:>7}{n+p:>7}   {comp}{flag}")
    print(f"  {'合计':<18}{tc:>7}{tp:>7}{tc+tp:>7}   （另排除地理 {n_excl} 题）")

    print()
    print("  === 合并后的可用性（两题库合计）===")
    both = [d for d in PROX14 if sum(cell.get(d, Counter()).values()) >= 20 and prox.get(d, 0) >= 20]
    only_c = [d for d in PROX14 if sum(cell.get(d, Counter()).values()) >= 20 and prox.get(d, 0) < 20]
    only_p = [d for d in PROX14 if sum(cell.get(d, Counter()).values()) < 20 and prox.get(d, 0) >= 20]
    print(f"    ✅ 两题库都有 >=20 题（可合并对比）: {len(both)} 个")
    for d in both:
        print(f"        {d:<18} C-Eval {sum(cell.get(d,Counter()).values()):>4} + ProX {prox.get(d,0):>4}"
              f" = {sum(cell.get(d,Counter()).values())+prox.get(d,0):>4}")
    print(f"    ⚠️ 只有 C-Eval 够: {only_c}")
    print(f"    ⚠️ 只有 ProX   够: {only_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
