"""Diversity scorer for evaluating result diversity and complementarity"""

from eval_gvm_vector_search.scorers.base import BaseScorer


class DiversityScorer(BaseScorer):
    """
    Evaluates whether result set is diverse and complementary.
    
    Ensures top results cover different angles/subtopics/approaches rather than being repetitive.
    """
    
    @property
    def name(self) -> str:
        return "diversity"
    
    @property
    def description(self) -> str:
        return "Evaluates whether result set is diverse and complementary, avoiding high repetition in top results"
    
    @property
    def instructions(self) -> str:
        return """3) Diversity（多樣性/互補性）

你是檢索結果品質評審。你的任務是評估「結果集合是否多樣且互補」，避免前排內容高度重複，並能涵蓋不同角度/子主題/立場/做法。

REQUEST:
{{ inputs }}

RETRIEVED RESULTS（按排名順序）:
{{ outputs }}

評估規則（請在腦中完成，不要輸出過程）：
- 只看"結果彼此之間"的互補性，不評估單篇是否好看
- 高分意味著：同一查詢下，結果提供不同子主題、不同切入角度（教學/案例/對比/避坑/背景/最新），而不是換標題重複同一段話
- 若 Top 5 來源/角度/結構高度同質（像同一篇改寫、同媒體洗稿、同結論重複），要扣分
- 多樣性不等於離題：如果為了多樣而塞入不相關內容，也要扣分

分數定義（1-5）：
5 = 結果高度互補，能覆蓋多個相關子角度且不離題
4 = 有明顯互補性，少數重複
3 = 一半互補一半重複，或角度偏單一
2 = 高度同質化，重複明顯
1 = 幾乎全在重複或為多樣而離題

只輸出一個數字：1 / 2 / 3 / 4 / 5"""