-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- PPISP (Physically Plausible Image Signal Processing) SPG Launcher
--
-- Binds PPISP parameters and dispatches the compute shader for
-- USD RenderProduct post-processing.
--
-- NOTE: Uses flat parameter names matching USD inputs: attributes (UsdShade-compatible).

function ppispProcess(inputs, outputs, params)
    local in_rgba = inputs["HdrColor"]
    assert(in_rgba and in_rgba.rank == 2, "HdrColor input must be a 2D texture")

    -- LdrColor expects an RGBA8 output, even when the input is HdrColor.
    local height = in_rgba.shape[1]
    local width = in_rgba.shape[2]
    outputs["PPISPColor"] = slang.empty({height, width}, slang.uchar4)

    -- Pass params directly to preserve __fullName for shader reflection matching.
    local function getFloat2(name)
        local p = params[name]
        return p and slang.float2(p) or slang.float2(0.0, 0.0)
    end

    return slang.dispatch({
        stage = "compute",
        numthreads = { 16, 16, 1 },
        grid = { math.ceil(width / 16), math.ceil(height / 16), 1 },
        bind = {
            slang.ParameterBlock(
                -- Exposure
                slang.float(params["exposureOffset"] or 0.0),

                -- Vignetting R
                getFloat2("vignettingCenterR"),
                slang.float(params["vignettingAlpha1R"] or 0.0),
                slang.float(params["vignettingAlpha2R"] or 0.0),
                slang.float(params["vignettingAlpha3R"] or 0.0),

                -- Vignetting G
                getFloat2("vignettingCenterG"),
                slang.float(params["vignettingAlpha1G"] or 0.0),
                slang.float(params["vignettingAlpha2G"] or 0.0),
                slang.float(params["vignettingAlpha3G"] or 0.0),

                -- Vignetting B
                getFloat2("vignettingCenterB"),
                slang.float(params["vignettingAlpha1B"] or 0.0),
                slang.float(params["vignettingAlpha2B"] or 0.0),
                slang.float(params["vignettingAlpha3B"] or 0.0),

                -- Color latent offsets (4 control points)
                getFloat2("colorLatentBlue"),
                getFloat2("colorLatentRed"),
                getFloat2("colorLatentGreen"),
                getFloat2("colorLatentNeutral"),

                -- CRF R (defaults = identity: boundedSoftplus(0.013659,0.3)=1, sigmoid(0)=0.5)
                slang.float(params["crfToeR"] or 0.013659),
                slang.float(params["crfShoulderR"] or 0.013659),
                slang.float(params["crfGammaR"] or 0.378165),
                slang.float(params["crfCenterR"] or 0.0),

                -- CRF G
                slang.float(params["crfToeG"] or 0.013659),
                slang.float(params["crfShoulderG"] or 0.013659),
                slang.float(params["crfGammaG"] or 0.378165),
                slang.float(params["crfCenterG"] or 0.0),

                -- CRF B
                slang.float(params["crfToeB"] or 0.013659),
                slang.float(params["crfShoulderB"] or 0.013659),
                slang.float(params["crfGammaB"] or 0.378165),
                slang.float(params["crfCenterB"] or 0.0)
            ),
            slang.Texture2D(in_rgba),
            slang.RWTexture2D(outputs["PPISPColor"]),
        },
    })
end
