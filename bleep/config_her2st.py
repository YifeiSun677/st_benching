"""
BLEEP hyperparameters, adapted for her2st.

Every value here is BLEEP's own default from bowang-lab/BLEEP/config.py
unless flagged CHANGED. Keep it that way -- deviations must be
declarable in methods.
"""

# --- model ---------------------------------------------------------------
model_name = "resnet50"
image_embedding = 2048           # resnet50 pooled feature dim
spot_embedding = 833             # CHANGED: 3467 (liver HVG union) -> 833 panel
pretrained = True                # ImageNet
trainable = True                 # image encoder is fine-tuned, not frozen
temperature = 1.0

# projection head (shared config for image and spot branches)
num_projection_layers = 1
projection_dim = 256
dropout = 0.1

# --- optimisation --------------------------------------------------------
lr = 1e-3
weight_decay = 1e-3
patience = 2                     # unused: BLEEP comments out the LR scheduler
factor = 0.5                     # unused, ditto

# --- data ----------------------------------------------------------------
size = 224                       # patch is 224x224 centred on the spot
