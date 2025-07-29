# variables
MODEL_SIZE = "small"

# definitions
n_channels = 3

tiny_model_config = dict(
    in_nc=n_channels + 1,
    out_nc=n_channels,
    nc=[4, 8, 16, 32], 
    nb=4,
    act_mode="R",
    downsample_mode="strideconv",
    upsample_mode="convtranspose",
)

small_model_config = dict(
    in_nc=n_channels + 1,
    out_nc=n_channels,
    nc=[8, 16, 32, 64], 
    nb=4,
    act_mode="R",
    downsample_mode="strideconv",
    upsample_mode="convtranspose",
)

medium_model_config = dict(
    in_nc=n_channels + 1,
    out_nc=n_channels,
    nc=[16, 32, 64, 128], 
    nb=4,
    act_mode="R",
    downsample_mode="strideconv",
    upsample_mode="convtranspose",
)

large_model_config = dict(
    in_nc=n_channels + 1,
    out_nc=n_channels,
    nc=[32, 64, 128, 256], 
    nb=4,
    act_mode="R",
    downsample_mode="strideconv",
    upsample_mode="convtranspose",
)

model_pool = {
    "tiny": tiny_model_config,
    "small": small_model_config,
    "medium": medium_model_config,
    "large": large_model_config,
}


model_config = model_pool[MODEL_SIZE]
