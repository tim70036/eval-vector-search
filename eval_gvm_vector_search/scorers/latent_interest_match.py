"""Latent Interest Match scorer for evaluating implicit user needs"""

from eval_gvm_vector_search.scorers.base import BaseScorer


class LatentInterestMatchScorer(BaseScorer):
    """
    Evaluates whether results match the user's underlying implicit interests and motivations.
    
    Focuses on capturing what the user really wants to know, not just surface-level keywords.
    """
    
    @property
    def name(self) -> str:
        return "latent_interest_match"
    
    @property
    def description(self) -> str:
        return "Evaluates whether results match user's underlying implicit interests and motivations"
    
    @property
    def instructions(self) -> str:
        return """2) Latent Interest Match（潛在興趣吻合）

你是推薦系統的評審。你的任務是評估「結果是否抓到使用者查詢背後真正想看的角度/動機」，讓他會覺得"這正是我在找的"。

REQUEST:
{{ inputs }}

RETRIEVED RESULTS（按排名順序）:
{{ outputs }}

評估規則（請在腦中完成，不要輸出過程）：
- 你要從查詢推測使用者背後可能的隱含需求（例如：想避坑、想省錢、想快速上手、想看結論、想看對比、想看案例、想看最新變動、想知道風險）
- 高分需：多數結果（尤其 Top 3）能命中這些隱含角度，而不是停留在表面定義或泛泛介紹
- 若結果只回答"這是什麼"，沒有回答"為什麼你會在意/你接下來要怎麼做"，就扣分
- 不評估是否誇大或是否可信（那是 credibility 指標）

分數定義（1-5）：
5 = Top 3 明顯命中查詢背後的核心關切點，角度精準且對胃口
4 = 多數命中隱含需求，少量結果偏表面
3 = 有部分命中，但不少內容仍停在泛泛層次
2 = 大多只貼表面關鍵字，沒抓到使用者真正想看的點
1 = 完全抓錯方向或沒有潛在興趣線索

只輸出一個數字：1 / 2 / 3 / 4 / 5"""