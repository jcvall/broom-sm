
import pandas as pd
import sys

import stats_tidy
from .broom_func.stats_tidy import stats_tidy
from .broom_func.stats_glance import stats_glance
from .broom_func.stats_augment import stats_augment
from .broom_func.boot_tidy import boot_tidy
from .broom_funcboot_glance import boot_glance
from .broom_funcboot_augment import boot_augment
from .broom_func.stats_power import stats_power
from .broom_func.stats_residual_plot import stats_residual_plot
from .broom_func.stats_ols_plot import stats_ols_plot
from .broom_func.stats_influence_plot import stats_influence_plot
from .broom_func.stats_chisquare_plot import stats_chisquare_plot
from .broom_func.stats_vif import stats_vif
from .broom_func.stats_conprob import stats_conprob
from .broom_func.bayes_boot import bayes_boot
from .broom_func.bayes_boot_plot import bayes_boot_plot
from .broom_func.stats_anova_tidy import stats_anova_tidy
from .broom_func.stats_kruskal_tidy import stats_kruskal_tidy
from .broom_func.stats_correlation_tidy import stats_correlation_tidy
from .broom_func.stats_formula import stats_formula

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "broom_sm"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
