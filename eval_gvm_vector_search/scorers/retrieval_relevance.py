"""Retrieval relevance scorer for evaluating search result relevance"""

from eval_gvm_vector_search.scorers.base import BaseScorer


class RetrievalRelevanceScorer(BaseScorer):
    """
    Evaluates whether retrieved results satisfy the user's query intent.
    
    Focus on Top 3 results with strict evaluation criteria.
    Uses Chinese language instructions for better alignment with target use case.
    """
    
    @property
    def name(self) -> str:
        return "retrieval_relevance"
    
    @property
    def description(self) -> str:
        return "Evaluates whether retrieved results satisfy user query intent (focuses on Top 3)"
    
    @property
    def instructions(self) -> str:
        return """
你是搜尋/推薦系統的嚴格評審。你要評估「檢索結果是否滿足使用者這個查詢的意圖」，並且重點看 Top 3。

REQUEST（使用者查詢）:
{{ inputs }}

RETRIEVED RESULTS（按排名順序）:
{{ outputs }}

評分步驟（請在腦中完成，不要輸出過程）：
1) 先判斷這個查詢屬於哪種意圖（例如：學習解釋、比較選擇、找清單/範例、找新聞/最新、解決問題/避坑、娛樂消遣）。
2) 逐一判斷每個結果是否「直接回應該意圖」，不要只看關鍵字重合。
3) Top 3 權重最高：如果 Top 3 有明顯偏題，即使後面有相關也要大幅扣分。
4) 覆蓋度：如果使用者意圖需要多面向（例如比較/避坑/步驟/案例），結果是否能互補涵蓋，而不是重複。

分數定義（1-5）：
5 = Top 3 幾乎都精準命中意圖，且整體涵蓋互補、缺口很小
4 = 大多命中意圖，Top 3 只有小瑕疵或有一篇略偏
3 = 有相關但不穩定：Top 3 有明顯缺口/偏題，或覆蓋面不足
2 = 多數偏題，只剩少量內容勉強可用
1 = 幾乎完全不相關

只輸出一個數字：1 / 2 / 3 / 4 / 5
"""
