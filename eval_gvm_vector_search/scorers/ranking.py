"""Ranking scorer for evaluating result ordering quality"""

from eval_gvm_vector_search.scorers.base import BaseScorer


class RankingScorer(BaseScorer):
    """
    Evaluates whether results are ranked according to defined value hierarchy.
    
    Prioritizes intent fit, latent interest match, engagement, and diversity.
    """
    
    @property
    def name(self) -> str:
        return "ranking"
    
    @property
    def description(self) -> str:
        return "Evaluates whether results are ranked according to defined value hierarchy"
    
    @property
    def instructions(self) -> str:
        return """5) Ranking（排序品質：依你定義的價值排序）

你是排序系統評審。你的任務是評估「目前排名是否把最該優先展示的內容放在前面」。

REQUEST:
{{ inputs }}

RETRIEVED RESULTS（按排名順序）:
{{ outputs }}

排序應遵守的價值順序（高到低）：
1) Intent Fit（意圖吻合）
2) Latent Interest Match（命中隱含關切）
3) Engagement（吸睛但不誤導）
4) Diversity（前排互補而非重複）
5) 明顯不可信/硬廣/誤導的結果應往後排

評估規則：
- 如果第 1 名或 Top 3 明顯偏題、抓錯意圖、或看起來很誤導 → 直接低分
- 如果高品質內容被埋到後面，而前面是同質重複或偏弱 → 扣分
- 如果前排彼此高度重複導致使用者看不出"不同收穫" → 扣分

分數定義（1-5）：
5 = Top 3 幾乎完美且互補，整體排序符合價值順序
4 = 大致合理，只有少量位置可微調
3 = 有明顯可改進：1-2 篇應上調/下調或前排重複
2 = 排序問題大：偏題在前、好內容被埋或誤導未下沉
1 = 幾乎亂排或反向排

只輸出一個數字：1 / 2 / 3 / 4 / 5"""