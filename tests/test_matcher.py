import pytest
from src.matcher.llm_evaluator import JobMatcher

def test_job_matcher_heuristic():
    matcher = JobMatcher()
    
    # Relevant quant role
    quant_jd = """
    Looking for a Quantitative Researcher with strong Python and C++ skills to develop
    statistical arbitrage alphas, backtest trading strategies, and analyze market microstructure.
    """
    res = matcher.evaluate_job("Quant Trader", "Alpha Capital", quant_jd)
    assert res.fit_score >= 70
    assert res.verdict == "MATCHED"
    assert len(res.fit_reasons) > 0

    # Non-relevant role
    irrelevant_jd = """
    We need an HR Manager and Recruiter with 10+ years of corporate compliance and payroll experience.
    Must have full stack Java experience and sales background.
    """
    res_irr = matcher.evaluate_job("Human Resources Director", "Corp XYZ", irrelevant_jd)
    assert res_irr.fit_score < 60
    assert res_irr.verdict == "SKIPPED"
