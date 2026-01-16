"""Engagement scorer for evaluating clickability without misleading"""

from eval_gvm_vector_search.scorers.base import BaseScorer


class EngagementScorer(BaseScorer):
    """
    Evaluates whether results are eye-catching and clickable without being misleading.
    
    Focuses on title/abstract appeal while avoiding clickbait or false promises.
    """
    
    @property
    def name(self) -> str:
        return "engagement"
    
    @property
    def description(self) -> str:
        return "Evaluates whether results are eye-catching and clickable without being misleading"
    
    @property
    def instructions(self) -> str:
        return """4) Engagement（吸睛但不誤導）

你是內容推薦評審。你的任務是評估「對該查詢有需求的使用者，看到標題/摘要是否會想點」，但必須"吸睛且不誤導"。

REQUEST:
{{ inputs }}

RETRIEVED RESULTS（按排名順序）:
{{ outputs }}

評估規則（請在腦中完成，不要輸出過程）：
- 只根據標題/摘要判斷點擊動機（不要假設你已讀完整文）
- 高分需要同時滿足：
  1) 有清楚價值主張或好奇缺口（點進去我會得到什麼）
  2) 具體（數字、清單、案例、對比、避坑、結論導向）
  3) 與查詢意圖一致（不是硬蹭流量）
- 重大扣分紅線（屬於誤導式吸睛）：
  - 誇大恐嚇、暗示必賺/必瘦/必翻盤、用情緒勒索、斷章取義
  - 看起來像農場文/洗稿/硬廣，或標題與摘要自相矛盾

分數定義（1-5）：
5 = Top 3 非常想點：具體、有亮點、價值清楚，且沒有明顯誤導味
4 = 整體有吸引力，少數偏普通或略有行銷腔
3 = 有些可點但多數平淡，或吸睛與可信之間搖擺
2 = 大多不想點（空泛/廣告感/套路），或誤導風險偏高
1 = 幾乎不會點，明顯標題黨或低質農場文感

只輸出一個數字：1 / 2 / 3 / 4 / 5"""