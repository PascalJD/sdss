# utils/eval_selector.py
def get_eval_fn(alg_name: str):
    if alg_name == "dis":
        from algorithms.dis.dis_eval import dis_eval
        return dis_eval
    elif alg_name == "pis":
        from algorithms.pis.pis_eval import pis_eval
        return pis_eval
    elif alg_name == "dds":
        from algorithms.dds.dds_eval import dds_eval
        return dds_eval
    elif alg_name == "sdss_vp":
        from algorithms.sdss_vp.eval import run_eval_sdss_vp
        return run_eval_sdss_vp
    raise ValueError(f"No evaluation entrypoint for algorithm '{alg_name}'.")