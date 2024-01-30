import inspect
import sys

def cluster_info(module_name):
    """Collect information of all metrics, including:

        - ``metric_need``: Information needed to calculate this metric, the combination of ``rec.items, rec.topk,
          rec.meanrank, rec.score, data.num_items, data.num_users, data.count_items, data.count_users, data.label``.
        - ``metric_type``: Whether the scores required by metric are grouped by user, range in ``EvaluatorType.RANKING``
          and ``EvaluatorType.VALUE``.
        - ``smaller``: Whether the smaller metric value represents better performance,
          range in ``True`` and ``False``, default to ``False``.

    Args:
        module_name (str): the name of module ``recbole.evaluator.metrics``.

    Returns:
        dict: Three dictionaries containing the above information
        and a dictionary matching metric names to metric classes.
    """
    smaller_m = []
    m_dict, m_info, m_types = {}, {}, {}
    metric_class = inspect.getmembers(
        sys.modules[module_name],
        lambda x: inspect.isclass(x) and x.__module__ == module_name,
    )
    for name, metric_cls in metric_class:
        name = name.lower()
        m_dict[name] = metric_cls
        if hasattr(metric_cls, "metric_need"):
            m_info[name] = metric_cls.metric_need
        else:
            raise AttributeError(f"Metric '{name}' has no attribute [metric_need].")
        if hasattr(metric_cls, "metric_type"):
            m_types[name] = metric_cls.metric_type
        else:
            raise AttributeError(f"Metric '{name}' has no attribute [metric_type].")
        if metric_cls.smaller is True:
            smaller_m.append(name)
    return smaller_m, m_info, m_types, m_dict


metric_module_name = "talkingface.evaluator.metrics"
smaller_metrics, metric_information, metric_types, metrics_dict = cluster_info(
    metric_module_name
)