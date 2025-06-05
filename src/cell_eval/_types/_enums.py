import enum


class DESortBy(enum.Enum):
    """Sorting options for differential expression results."""

    FOLD_CHANGE = "log2_fold_change"
    ABS_FOLD_CHANGE = "abs_log2_fold_change"
    PVALUE = "p_value"
    FDR = "fdr"


class MetricType(enum.Enum):
    """Types of metrics supported by the registry."""

    DE = "de"
    ANNDATA_PAIR = "anndata_pair"
