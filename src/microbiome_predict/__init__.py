"""
microbiome_predict
==================
Predictive modeling layer for the microbiome disease-prediction project.

It consumes the standardized sample x feature abundance matrix produced by the
upstream Snakemake preprocessing pipeline (``results/matrix/
species_abundance_matrix.csv``) and turns it into clinically interpretable
predictions.

Implemented components (mapping to the project spec, Sections 3-7):

* ``EnsembleDiseaseClassifier`` — Model 3: a soft-voting ensemble
  (Random Forest + SVM + Gradient Boosting + XGBoost) for specific-disease
  classification, with compositionally-aware feature handling (CLR).
* ``WellnessIndex`` — Model 2: a transparent, GMWI2-style health index that
  separates healthy from diseased microbiome profiles.
* ``PrognosticModel`` — Model 4: a Cox proportional-hazards survival model for
  future-disease (time-to-event) risk forecasting.
* ``uncertainty`` — reject option, prediction intervals and per-sample
  confidence scores (Section 5).
* ``interpret`` — global (permutation) and local (occlusion / SHAP) feature
  attribution (Section 4/6).
* ``validation`` — cross-validation and selective-prediction metrics.
* ``report`` — clinician-facing HTML report generation (Section 4).

NOTE ON ACCURACY CLAIMS: as the spec itself acknowledges, guaranteeing
98-100% accuracy for disease prediction from microbiome data alone is not
biologically realistic. Every prediction here is a calibrated *probability*
with an explicit uncertainty estimate and an optional reject ("indeterminate")
band — never a deterministic diagnosis.
"""

from .features import PrevalenceFilter, RelativeAbundance, CLRTransformer
from .models.ensemble import EnsembleDiseaseClassifier
from .models.wellness_index import WellnessIndex
from .models.survival import PrognosticModel
from .models.interpretable import InterpretableRuleClassifier, MMETHANEAdapter
from .models.deepsurv import DeepSurvModel
from .bundle import TrainedBundle
from . import data, uncertainty, interpret, validation, report, integration, pdf, ingest, classify

__version__ = "0.1.0"

__all__ = [
    "PrevalenceFilter",
    "RelativeAbundance",
    "CLRTransformer",
    "EnsembleDiseaseClassifier",
    "WellnessIndex",
    "PrognosticModel",
    "InterpretableRuleClassifier",
    "MMETHANEAdapter",
    "DeepSurvModel",
    "TrainedBundle",
    "data",
    "uncertainty",
    "interpret",
    "validation",
    "report",
    "integration",
    "pdf",
    "ingest",
    "classify",
]
