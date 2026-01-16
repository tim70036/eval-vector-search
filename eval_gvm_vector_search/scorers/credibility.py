"""Credibility scorer for evaluating non-misleading and trustworthy content"""

from eval_gvm_vector_search.scorers.base import BaseScorer


class CredibilityScorer(BaseScorer):
    """
    Evaluates whether results are credible and non-misleading.
    
    Avoids clickbait, false promises, low-quality content farms, and misleading information.
    """
    
    @property
    def name(self) -> str:
        return "credibility"
    
    @property
    def description(self) -> str:
        return "Evaluates whether results are credible and non-misleading, avoiding clickbait and false promises"
    
    @property
    def instructions(self) -> str:
        return """6) Non-misleading / Credibility（不誤導與可信度）

你是內容風險與可信度評審。你的任務是評估「這組結果是否可信、不誤導」，避免把點擊率建立在錯誤、誇大或硬廣上。

REQUEST:
{{ inputs }}

RETRIEVED RESULTS（按排名順序）:
{{ outputs }}

評估規則（請在腦中完成，不要輸出過程）：
- 高分代表：多數結果看起來資訊可靠、論述自洽、標題與摘要一致、沒有明顯操弄或硬廣味
- 重大扣分紅線（任一在 Top 3 出現就強扣）：
  - 標題黨：誇大、恐嚇、暗示保證收益/療效/翻身
  - 明顯錯誤或常識性矛盾（僅從摘要即可判斷）
  - 低品質農場文、洗稿、硬廣導購但偽裝成科普
  - 標題與摘要落差大（疑似騙點擊）
- 不要求學術引用，但要看得出"像一篇負責任的文章"

分數定義（1-5）：
5 = 幾乎沒有誤導風險，結果整體可信且自洽
4 = 大多可信，少數略有行銷腔或不夠嚴謹
3 = 可信度不穩，有幾篇看起來像套路文/資訊偏薄
2 = 誤導/硬廣/農場文風險明顯，Top 3 可能踩雷
1 = 明顯不可信或高度誤導，整組結果風險很高

只輸出一個數字：1 / 2 / 3 / 4 / 5"""