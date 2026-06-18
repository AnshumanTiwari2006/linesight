"""
LineSight visual identity — single source of truth for all plot colors.
All visualization files import from here. Never hardcode hex values elsewhere.
"""

# Primary data and model colors
DATA_POINTS      = "#888888"   # neutral gray for scatter points
FIT_LINE         = "#1a6fcc"   # blue — the model's line
RESIDUAL_POS     = "#e05252"   # red — point is above the line (positive residual)
RESIDUAL_NEG     = "#5280e0"   # blue — point is below the line (negative residual)

# Classification colors
CLASS_0          = "#4a90d9"   # blue for negative class
CLASS_1          = "#d9534a"   # red for positive class
DECISION_BOUNDARY = "#111111"  # dark line for boundary

# Loss and optimization
LOSS_LINE        = "#1a6fcc"   # blue for loss curve
LOSS_FINAL_DOT   = "#e05252"   # red dot at final loss value
GRADIENT_PATH    = "#e05252"   # red path on loss surface

# Regularization comparison
COLOR_LINEAR     = "#1a6fcc"   # blue
COLOR_RIDGE      = "#2a9d4e"   # green
COLOR_LASSO      = "#e05252"   # red
COLOR_ELASTICNET = "#7047c4"   # purple

# Feature importance
BAR_POSITIVE     = "#1a6fcc"   # blue for positive coefficients
BAR_NEGATIVE     = "#e05252"   # red for negative coefficients
BAR_ZERO         = "#cccccc"   # gray for zeroed-out (Lasso)

# Annotation and warning
WARNING_COLOR    = "#e05252"   # red for convergence warnings
ANNOTATION_COLOR = "#555555"   # gray for equation text overlays

# Background and grid
SPINE_COLOR      = "#dddddd"
GRID_COLOR       = "#f0f0f0"

# Surface plots
SURFACE_CMAP     = "Blues"
BOUNDARY_CMAP    = "RdBu_r"
